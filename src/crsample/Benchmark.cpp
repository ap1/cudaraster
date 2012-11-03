/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Benchmark.hpp"
#include "Shaders.hpp"
#include "3d/Mesh.hpp"
#include "base/Timer.hpp"
#include "base/Random.hpp"

using namespace FW;

//------------------------------------------------------------------------

Benchmark::Benchmark(void)
:   m_numVertices   (0),
    m_numMaterials  (0),
    m_numTriangles  (0),
    m_enableLerp    (true),
    m_colorBuffer   (NULL),
    m_depthBuffer   (NULL)
{
    // Initialize CUDA compiler.

    m_cudaCompiler.setSourceFile("src/crsample/Shaders.cu");
    m_cudaCompiler.addOptions("-use_fast_math");
    m_cudaCompiler.include("src/framework");
    m_cudaCompiler.include("src/cudaraster");

    // Initialize GL.

    m_window.setVisible(false);
    m_window.getGL();

    glGenRenderbuffers(1, &m_glDrawColorRenderbuffer);
    glGenRenderbuffers(1, &m_glDrawDepthRenderbuffer);
	glGenRenderbuffers(1, &m_glResolveColorRenderbuffer);
    glGenFramebuffers(1, &m_glDrawFramebuffer);
	glGenFramebuffers(1, &m_glResolveFramebuffer);

    GLContext::checkErrors();
}

//------------------------------------------------------------------------

Benchmark::~Benchmark(void)
{
    delete m_colorBuffer;
    delete m_depthBuffer;

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteRenderbuffers(1, &m_glDrawColorRenderbuffer);
    glDeleteRenderbuffers(1, &m_glDrawDepthRenderbuffer);
    glDeleteRenderbuffers(1, &m_glResolveColorRenderbuffer);
    glDeleteFramebuffers(1, &m_glDrawFramebuffer);
    glDeleteFramebuffers(1, &m_glResolveFramebuffer);
}

//------------------------------------------------------------------------

void Benchmark::log(const String& fileName, const String& msg)
{
    if (fileName.getLength())
        pushLogFile(fileName, true);

    printf("%s", msg.getPtr());

    if (fileName.getLength())
        popLogFile();

    failIfError();
}

//------------------------------------------------------------------------

String Benchmark::appendFileName(const String& fileName, const String& postfix)
{
    if (!fileName.getLength())
        return "";

    int dotIdx = fileName.lastIndexOf('.');
    if (dotIdx == -1)
        dotIdx = fileName.getLength();

    return fileName.substring(0, dotIdx) + postfix + fileName.substring(dotIdx);
}

//------------------------------------------------------------------------

void Benchmark::loadMesh(const String& fileName)
{
    // Import mesh.

    MeshBase* mesh = importMesh(fileName);
    failIfError();

    // Fix geometry.

    mesh->clean();                // Remove degenerates, etc.
    mesh->dupVertsPerSubmesh();   // Makes it possible to assign materials to vertices unambiguously.
    mesh->fixMaterialColors();    // Come up with reasonable material colors for Gouraud shading.

    // Convert to PNT.

    Mesh<VertexPNT> convMesh(*mesh);
    m_numVertices = convMesh.numVertices();
    m_numMaterials = convMesh.numSubmeshes();
    m_numTriangles = convMesh.numTriangles();

    // Allocate buffers.

    m_inputVertices.resizeDiscard       (m_numVertices  * sizeof(InputVertex));
    m_shadedVertices.resizeDiscard      (m_numVertices  * sizeof(GouraudVertex));
    m_materials.resizeDiscard           (m_numMaterials * sizeof(Material));
    m_vertexIndices.resizeDiscard       (m_numTriangles * sizeof(Vec3i));
    m_vertexMaterialIdx.resizeDiscard   (m_numVertices  * sizeof(S32));

    // Copy vertex attributes.

    InputVertex* inputVertexPtr = (InputVertex*)m_inputVertices.getMutablePtr();
    for (int i = 0; i < m_numVertices; i++)
    {
        const VertexPNT& in = convMesh.vertex(i);
        InputVertex& out = inputVertexPtr[i];
        out.modelPos    = in.p;
        out.modelNormal = in.n;
        out.texCoord    = in.t;
    }

    // Copy vertex indices.

    Vec3i* vertexIndexPtr = (Vec3i*)m_vertexIndices.getMutablePtr();
    for (int i = 0; i < m_numMaterials; i++)
        for (int j = 0; j < convMesh.indices(i).getSize(); j++)
            *vertexIndexPtr++ = convMesh.indices(i)[j];

    // Setup materials.

    Material* materialPtr = (Material*)m_materials.getMutablePtr();
    for (int i = 0; i < m_numMaterials; i++)
    {
        const MeshBase::Material& in = convMesh.material(i);
        Material& out = materialPtr[i];
        out.diffuseColor    = Vec4f(in.diffuse.getXYZ(), 0.5f); // force alpha to 0.5
        out.specularColor   = in.specular * 0.5f;
        out.glossiness      = in.glossiness;
    }

    // Setup material indices.

    S32* vertexMaterialIdxPtr = (S32*)m_vertexMaterialIdx.getMutablePtr();
    for (int i = 0; i < m_numMaterials; i++)
    for (int j = 0; j < convMesh.indices(i).getSize(); j++)
    for (int k = 0; k < 3; k++)
        vertexMaterialIdxPtr[convMesh.indices(i)[j][k]] = i;
}

//------------------------------------------------------------------------

void Benchmark::initPipe(const Vec2i& resolution, int numSamples, U32 renderModeFlags, Blend blend, int profilingMode)
{
    // Setup GL framebuffer objects.

    int glSamples = (numSamples > 1) ? numSamples : 0;

    glBindRenderbuffer(GL_RENDERBUFFER, m_glDrawColorRenderbuffer);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, glSamples, GL_RGBA, resolution.x, resolution.y);
    glBindRenderbuffer(GL_RENDERBUFFER, m_glDrawDepthRenderbuffer);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, glSamples, GL_DEPTH_COMPONENT, resolution.x, resolution.y);
	glBindRenderbuffer(GL_RENDERBUFFER, m_glResolveColorRenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, resolution.x, resolution.y);

    glBindFramebuffer(GL_FRAMEBUFFER, m_glDrawFramebuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_glDrawColorRenderbuffer);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_glDrawDepthRenderbuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, m_glResolveFramebuffer);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_glResolveColorRenderbuffer);

    GLContext::checkErrors();

    // Setup GL state.

    glBindFramebuffer(GL_FRAMEBUFFER, m_glDrawFramebuffer);
    glViewport(0, 0, resolution.x, resolution.y);

    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glShadeModel(GL_FLAT);
    glDisable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    if ((renderModeFlags & RenderModeFlag_EnableDepth) != 0)
        glEnable(GL_DEPTH_TEST);

    m_enableLerp = ((renderModeFlags & RenderModeFlag_EnableLerp) != 0);
    if (m_enableLerp)
        glShadeModel(GL_SMOOTH);

    const char* blendShader;
    switch (blend)
    {
    case Blend_Replace:     blendShader = "BlendReplace"; break;
    case Blend_SrcOver:     blendShader = "BlendSrcOver"; glEnable(GL_BLEND); break;
    case Blend_DepthOnly:   blendShader = "BlendDepthOnly"; glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); break;
    default:                FW_ASSERT(false); return;
    }

    GLContext::checkErrors();

    // Create surfaces.

    delete m_colorBuffer;
    delete m_depthBuffer;
    m_colorBuffer = new CudaSurface(resolution, CudaSurface::Format_R8_G8_B8_A8, numSamples);
    m_depthBuffer = new CudaSurface(resolution, CudaSurface::Format_Depth32, numSamples);

    // Compile CUDA code.

    m_cudaCompiler.clearDefines();
    m_cudaCompiler.define("SAMPLES_LOG2", m_colorBuffer->getSamplesLog2());
    m_cudaCompiler.define("RENDER_MODE_FLAGS", renderModeFlags);
    m_cudaCompiler.define("BLEND_SHADER", blendShader);
    m_cudaCompiler.define("CR_PROFILING_MODE", profilingMode);
    m_cudaModule = m_cudaCompiler.compile();

    // Setup CudaRaster.

    m_cudaRaster.setSurfaces(m_colorBuffer, m_depthBuffer);
    m_cudaRaster.setPixelPipe(m_cudaModule, "PixelPipe_gouraud");
}

//------------------------------------------------------------------------

void Benchmark::runVertexShader(const String& camera)
{
    // Decode camera signature.

    m_cameraCtrl.decodeSignature(camera);
    Mat4f posToCamera = m_cameraCtrl.getWorldToCamera();
    Mat4f projection = Mat4f::fitToView(-1.0f, 2.0f, Vec2f(m_colorBuffer->getSize())) * m_cameraCtrl.getCameraToClip();
    failIfError();

    // Set globals.

    Constants& c = *(Constants*)m_cudaModule->getGlobal("c_constants").getMutablePtrDiscard();
    c.posToClip             = projection * posToCamera;
    c.posToCamera           = posToCamera;
    c.normalToCamera        = transpose(invert(posToCamera.getXYZ()));
    c.materials             = m_materials.getCudaPtr();
    c.vertexMaterialIdx     = m_vertexMaterialIdx.getCudaPtr();
    c.triangleMaterialIdx   = NULL;

    // Run vertex shader.

    m_cudaModule->getKernel("vertexShader_gouraud").setParams(m_inputVertices, m_shadedVertices, m_numVertices).launch(m_numVertices);
}

//------------------------------------------------------------------------

Benchmark::Result Benchmark::runSingle(Renderer renderer, int numBatches, const String& imageFileName)
{
    int glWarmupPasses = 10;
    int glBenchmarkPasses = 10;
    GLContext* gl = m_window.getGL();
    Result r;

    // CudaRaster.

    if (renderer == Renderer_CudaRaster)
    {
        m_cudaRaster.deferredClear(Vec4f(0.2f, 0.4f, 0.8f, 1.0f));
        m_cudaRaster.setVertexBuffer(&m_shadedVertices, 0);
    
        for (int i = 0; i < numBatches; i++)
        {
            int firstTri = (int)((S64)m_numTriangles * i / numBatches);
            int numTris = (int)((S64)m_numTriangles * (i + 1) / numBatches) - firstTri;
            if (numTris == 0)
                continue;

            m_cudaRaster.setIndexBuffer(&m_vertexIndices, firstTri * sizeof(Vec3i), numTris);
            m_cudaRaster.drawTriangles();

            CudaRaster::Stats s = m_cudaRaster.getStats();
            r.totalTime     += s.setupTime + s.binTime + s.coarseTime + s.fineTime;
            r.setupTime     += s.setupTime;
            r.binTime       += s.binTime;
            r.coarseTime    += s.coarseTime;
            r.fineTime      += s.fineTime;
            r.profilingInfo += m_cudaRaster.getProfilingInfo();
        }

        m_colorBuffer->resolveToScreen(gl);
    }

    // OpenGL.

    if (renderer == Renderer_OpenGL)
    {
        GouraudVertex* vtx = NULL;
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_vertexIndices.getGLBuffer());

        // Interpolation enabled => use custom shaders for better perf.

        if (m_enableLerp)
        {
            static const char* progId = "Benchmark::gouraud";
            GLContext::Program* prog = gl->getProgram(progId);
            if (!prog)
            {
                prog = new GLContext::Program(
                    "#version 120\n"
                    FW_GL_SHADER_SOURCE(
                        attribute vec4 positionAttrib;
                        attribute vec4 colorAttrib;
                        centroid varying vec4 colorVarying;
                        void main()
                        {
                            gl_Position = positionAttrib;
                            colorVarying = colorAttrib;
                        }
                    ),
                    "#version 120\n"
                    FW_GL_SHADER_SOURCE(
                        centroid varying vec4 colorVarying;
                        void main()
                        {
                            gl_FragColor = colorVarying;
                        }
                    ));
                gl->setProgram(progId, prog);
            }

            prog->use();
            gl->setAttrib(prog->getAttribLoc("positionAttrib"), 4, GL_FLOAT, sizeof(*vtx), m_shadedVertices, (int)&vtx->clipPos);
            gl->setAttrib(prog->getAttribLoc("colorAttrib"), 4, GL_FLOAT, sizeof(*vtx), m_shadedVertices, (int)&vtx->color);
        }

        // Interpolation disabled => use fixed-function shaders.

        else
        {
            glUseProgram(0);
            glBindBuffer(GL_ARRAY_BUFFER, m_shadedVertices.getGLBuffer());
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);
            glVertexPointer(4, GL_FLOAT, sizeof(*vtx), &vtx->clipPos);
            glColorPointer(4, GL_FLOAT, sizeof(*vtx), &vtx->color);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
        }

        // Warm up.

        glFinish();
        for (int pass = 0; pass < glWarmupPasses; pass++)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glDrawElements(GL_TRIANGLES, m_numTriangles * 3, GL_UNSIGNED_INT, NULL);
            glFinish();
        }

        // Benchmark.

        Timer timer(true);
        for (int pass = 0; pass < glBenchmarkPasses; pass++)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            for (int i = 0; i < numBatches; i++)
            {
                int firstTri = (int)((S64)m_numTriangles * i / numBatches);
                int numTris = (int)((S64)m_numTriangles * (i + 1) / numBatches) - firstTri;
                if (numTris != 0)
                {
                    glDrawElements(GL_TRIANGLES, numTris * 3, GL_UNSIGNED_INT, (void*)(firstTri * sizeof(Vec3i)));
                    glFinish();
                }
            }
        }
        r.totalTime += timer.getElapsed() / (F32)glBenchmarkPasses;

        // Clean up.

        gl->resetAttribs();
        glPopClientAttrib();
    }

    // Export image.

    if (imageFileName.getLength())
    {
        const Vec2i& resolution = m_colorBuffer->getSize();
        Image image(resolution);

		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_glResolveFramebuffer);
		glBlitFramebuffer(0, 0, resolution.x, resolution.y, 0, 0, resolution.x, resolution.y, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		glBindFramebuffer(GL_FRAMEBUFFER, m_glResolveFramebuffer);
        glReadPixels(0, 0, resolution.x, resolution.y, GL_RGBA, GL_UNSIGNED_BYTE, image.getMutablePtr());
		glBindFramebuffer(GL_FRAMEBUFFER, m_glDrawFramebuffer);
        GLContext::checkErrors();

        image.flipY();
        exportImage(imageFileName, &image);
        failIfError();
    }

    GLContext::checkErrors();
    return r;
}

//------------------------------------------------------------------------

Benchmark::Result Benchmark::runCameras(Renderer renderer, int numBatches, const String& imageFileName, const Array<String>& cameras)
{
    Result r;
    for (int i = 0; i < cameras.getSize(); i++)
    {
        runVertexShader(cameras[i]);
        r += runSingle(renderer, numBatches, appendFileName(imageFileName, sprintf("_cam%d", i)));
    }
    r *= 1.0f / (F32)cameras.getSize();
    return r;
}

//------------------------------------------------------------------------

void Benchmark::typeSingle(const Params& p)
{
    initPipe(p.resolution, p.numSamples, p.renderModeFlags, p.blend, ProfilingMode_Default);
    Result r = runCameras(p.renderer, 1, p.imageFileName, p.cameras);
    log(p.logFileName, sprintf("\t%.2fms\n", r.totalTime * 1.0e3f));
}

//------------------------------------------------------------------------

void Benchmark::typeResolution(const Params& p)
{
    log(p.logFileName, "\n");
    for (int scale = -1; scale <= 1; scale++)
    {
        Vec2i res = max(p.resolution << max(scale, 0) >> max(-scale, 0), Vec2i(1));
        String resStr = sprintf("%dx%d", res.x, res.y);
        String image = appendFileName(p.imageFileName, String("_") + resStr);
        initPipe(res, p.numSamples, p.renderModeFlags, p.blend, ProfilingMode_Default);
        Result r = runCameras(p.renderer, 1, image, p.cameras);
        log(p.logFileName, sprintf("\t%-12s%.2fms\n", resStr.getPtr(), r.totalTime * 1.0e3f));
    }
    log(p.logFileName, "\n");
}

//------------------------------------------------------------------------

void Benchmark::typeRenderMode(const Params& p)
{
    static const struct
    {
        const char* title;
        const char* tag;
        U32         flags;
        Blend       blend;
    } s_modes[] =
    {
        { "Depth, no blend",        "default",      RenderModeFlag_EnableDepth | RenderModeFlag_EnableLerp, Blend_Replace },
        { "No depth, alpha blend",  "blend",        RenderModeFlag_EnableLerp,                              Blend_SrcOver },
        { "Depth only",             "depthonly",    RenderModeFlag_EnableDepth,                             Blend_DepthOnly },
    };

    String msg;
    msg += "\n";
    msg += sprintf("\t%-24s", "RenderMode / MSAA");
    for (int numSamples = 1; numSamples <= 8; numSamples *= 2)
        msg += sprintf("%-8d", numSamples);
    msg += "\n";
    msg += sprintf("\t%-24s", "---");
    for (int numSamples = 1; numSamples <= 8; numSamples *= 2)
        msg += sprintf("%-8s", "---");
    msg += "\n";

    for (int mode = 0; mode < (int)FW_ARRAY_SIZE(s_modes); mode++)
    {
        msg += sprintf("\t%-24s", s_modes[mode].title);
        for (int numSamples = 1; numSamples <= 8; numSamples *= 2)
        {
            printf("Running %s / %dspp...\n", s_modes[mode].tag, numSamples);
            String image = appendFileName(p.imageFileName, sprintf("_%s_%dspp", s_modes[mode].tag, numSamples));
            initPipe(p.resolution, numSamples, s_modes[mode].flags, s_modes[mode].blend, ProfilingMode_Default);
            Result r = runCameras(p.renderer, 1, image, p.cameras);
            msg += sprintf("%-8.2f", r.totalTime * 1.0e3f);
        }
        msg += "\n";
    }
    msg += "\n";
    log(p.logFileName, msg);
}

//------------------------------------------------------------------------

void Benchmark::typeProfile(const Params& p)
{
    Result r;
    for (int mode = ProfilingMode_First; mode <= ProfilingMode_Last; mode++)
    {
        initPipe(p.resolution, p.numSamples, p.renderModeFlags, p.blend, mode);
        runVertexShader(p.cameras[0]);
        r += runSingle(Renderer_CudaRaster, 1, p.imageFileName);
    }
    log(p.logFileName, r.profilingInfo.getPtr());
}

//------------------------------------------------------------------------

void Benchmark::typeSynthetic(const Params& p)
{
    int numTicks        = 161;
    F32 maxArea         = 40.0f;
    F32 avgTriangles    = 1 << 20;
    int patchSize       = 8;
    int minLayers       = 16;

    Random rnd(123);
    Array<Vec4f> vertices;
    Array<Vec3i> indices;

    initPipe(p.resolution, p.numSamples, p.renderModeFlags, p.blend, ProfilingMode_Default);

    log(p.logFileName, "\n");
    log(p.logFileName, sprintf("\t%-8s%-8s%-8s%-8s%-8s%-8s\n", "Area", "Setup", "Bin", "Coarse", "Fine", "Total"));
    log(p.logFileName, sprintf("\t%-8s%-8s%-8s%-8s%-8s%-8s\n", "---", "---", "---", "---", "---", "---"));

    for (int tick = 0; tick < numTicks; tick++)
    {
        // Zero or small area => output backfacing triangles.

        F32 avgArea = (F32)tick / (F32)(numTicks - 1) * maxArea;
        if (avgArea < 1.0e-3f)
            avgArea = -1.0f;

        // Select parameters.

        F32 shrink = sqrt(abs(avgArea) * 2.0f);
        F32 trisPerLayer = ((F32)p.resolution.x - shrink) * ((F32)p.resolution.y - shrink) / abs(avgArea);
        int patchStep = max((int)(sqrt(trisPerLayer / avgTriangles * (F32)minLayers) * (F32)patchSize) + 1, patchSize);
        int numLayers = max((int)(avgTriangles / trisPerLayer * (F32)sqr(patchStep) / (F32)sqr(patchSize) + 0.5f), minLayers);

        // Generate each layer.

        vertices.clear();
        indices.clear();
        for (int layer = 0; layer < numLayers; layer++)
        {
            F32 z = 1.0f - ((F32)layer + 0.5f) / (F32)numLayers * 2.0f;
            F32 angle = rnd.getF32(0.0f, FW_PI * 2.0f);

            // Form grid-to-clip matrix.

            Mat3f shear;
            shear.m01 = 0.5f;
            shear.m11 = sqrt(0.75f);
            shear.m22 = sqrt(shear.m00 * shear.m11 * 0.5f);

            Mat3f rotate;
            rotate.m00 = cos(angle);
            rotate.m01 = sin(angle);
            rotate.m11 = cos(angle);
            rotate.m10 = -sin(angle);

            Mat3f xform = Mat3f::scale(Vec2f(sqrt(abs(avgArea)) * 2.0f) / Vec2f(p.resolution)) *
                rotate * shear * Mat3f::translate(rnd.getVec2f() * (F32)patchStep);

            if (avgArea < 0.0f)
                xform.col(0) *= -1.0f;

            // Compute AABB of the viewport in grid-space.

            Mat3f inv = invert(xform);
            Vec3f lof = inv.getCol(2) - abs(inv.getCol(0)) - abs(inv.getCol(1));
            Vec3f hif = inv.getCol(2) + abs(inv.getCol(0)) + abs(inv.getCol(1));
            Vec2i lo  = Vec2i((int)ceil(lof.x), (int)ceil(lof.y));
            Vec2i hi  = Vec2i((int)floor(hif.x), (int)floor(hif.y));

            // Process patches that intersect the viewport.

            for (int py = lo.y; py <= hi.y; py += patchStep)
            for (int px = lo.x; px <= hi.x; px += patchStep)
            {
                // Check corners against viewport.

                Vec2f c0 = xform * Vec2f((F32)(px + 0), (F32)(py + 0));
                Vec2f c1 = xform * Vec2f((F32)(px + patchSize), (F32)(py + 0));
                Vec2f c2 = xform * Vec2f((F32)(px + 0), (F32)(py + patchSize));
                Vec2f c3 = xform * Vec2f((F32)(px + patchSize), (F32)(py + patchSize));

                if (min(max(c0, c1, c2, c3)) < -1.0f || max(min(c0, c1, c2, c3)) > 1.0f)
                    continue;

                // Add vertices.

                int vbase = vertices.getSize();
                for (int y = 0; y <= patchSize; y++)
                for (int x = 0; x <= patchSize; x++)
                    vertices.add(Vec4f(xform * Vec2f((F32)(px + x), (F32)(py + y)), z, 1.0f));

                // Add triangles that are fully inside the viewport.

                for (int y = 0; y < patchSize; y++)
                for (int x = 0; x < patchSize; x++)
                {
                    Vec4i v = vbase + x + y * (patchSize + 1) + Vec4i(0, 1, patchSize + 1, patchSize + 2);
                    Vec2f p[4];
                    for (int i = 0; i < 4; i++)
                        p[i] = vertices[v[i]].getXY();

                    if (min(min(p[0], p[1], p[2])) >= -1.0f && max(max(p[0], p[1], p[2])) <= 1.0f)
                        indices.add(Vec3i(v[0], v[1], v[2]));

                    if (min(min(p[2], p[1], p[3])) >= -1.0f && max(max(p[2], p[1], p[3])) <= 1.0f)
                        indices.add(Vec3i(v[2], v[1], v[3]));
                }
            }
        }

        // Set geometry.

        m_shadedVertices.resizeDiscard(vertices.getSize() * sizeof(GouraudVertex));
        GouraudVertex* vertexPtr = (GouraudVertex*)m_shadedVertices.getMutablePtr();
        for (int i = 0; i < vertices.getSize(); i++)
        {
            vertexPtr[i].clipPos = vertices[i];
            vertexPtr[i].color = Vec4f(rnd.getVec3f(0.0f, 1.0f), 0.5f);
        }

        m_numTriangles = indices.getSize();
        m_vertexIndices.set(indices.getPtr(), indices.getNumBytes());

        // Benchmark.

        String image = appendFileName(p.imageFileName, sprintf("_tick%02d", tick));
        Result r = runSingle(p.renderer, 1, image);
        F32 coef = 1.0e9f / (F32)m_numTriangles;

        log(p.logFileName, sprintf("\t%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n",
            max(avgArea, 0.0f),
            r.setupTime * coef,
            r.binTime * coef,
            r.coarseTime * coef,
            r.fineTime * coef,
            r.totalTime * coef));
    }

    log(p.logFileName, "\n");
}

//------------------------------------------------------------------------

void Benchmark::typeBatch(const Params& p)
{
    initPipe(p.resolution, p.numSamples, p.renderModeFlags, p.blend, ProfilingMode_Default);

    log(p.logFileName, "\n");
    log(p.logFileName, sprintf("\t%-8s%-8s%-8s%-8s%-8s%-8s\n", "Batches", "Setup", "Bin", "Coarse", "Fine", "Total"));
    log(p.logFileName, sprintf("\t%-8s%-8s%-8s%-8s%-8s%-8s\n", "---", "---", "---", "---", "---", "---"));

    for (int batches = 1; batches <= 14; batches++)
    {
        String image = appendFileName(p.imageFileName, sprintf("_batch%02d", batches));
        Result r = runCameras(p.renderer, batches, image, p.cameras);

        log(p.logFileName, sprintf("\t%-8d%-8.2f%-8.2f%-8.2f%-8.2f%-8.2f\n",
            batches,
            r.setupTime * 1.0e3f,
            r.binTime * 1.0e3f,
            r.coarseTime * 1.0e3f,
            r.fineTime * 1.0e3f,
            r.totalTime * 1.0e3f));
    }

    log(p.logFileName, "\n");
}

//------------------------------------------------------------------------
