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

#include "App.hpp"
#include "Benchmark.hpp"
#include "base/Main.hpp"
#include "3d/Mesh.hpp"
#include "io/StateDump.hpp"

#include <stdio.h>
#include <conio.h>

using namespace FW;

//------------------------------------------------------------------------

static const char* const s_aboutText =
    "\"High-performance software rasterization on GPUs\",\n"
    "Samuli Laine and Tero Karras,\n"
    "Proc. High-Performance Graphics 2011\n"
    "\n"
    "http://code.google.com/p/cudaraster/\n"
    "\n"
    "Copyright 2010-2012 NVIDIA Corporation\n"
;

//------------------------------------------------------------------------

static const char* const s_commandHelpText =
    "\n"
    "Usage: crsample <mode> [options]\n"
    "\n"
    "Supported values for <mode>:\n"
    "\n"
    "   interactive         Run in interactive mode (default).\n"
    "   bench-single        Measure rendering time.\n"
    "   bench-resolution    Table 1: Benchmark different resolutions.\n"
    "   bench-rendermode    Table 2: Benchmark different rendering modes.\n"
    "   bench-profile       Table 3: Show detailed profiling information.\n"
    "   bench-synthetic     Figure 5: Synthetic test with equally-sized tris.\n"
    "   bench-batch         Figure 6: Break up the input into multiple batches.\n"
    "\n"
    "Benchmark options:\n"
    "\n"
    "   --mesh=<file.obj>   Mesh to benchmark.\n"
    "   --camera=\"<sig>\"    Camera signature. Can be specified multiple times.\n"
    "\n"
    "   --renderer=cuda     Benchmark CudaRaster (default).\n"
    "   --renderer=gl       Benchmark OpenGL.\n"
    "\n"
    "   --size=<w>x<h>      Resolution. Default is \"1024x768\".\n"
    "   --samples=<value>   Number of samples per pixel. Default is \"1\".\n"
    "   --depth=<1/0>       Enable depth test. Default is \"1\".\n"
    "   --lerp=<1/0>        Enable attribute interpolation. Default is \"1\".\n"
    "   --quads=<1/0>       Enable pixel-quad rendering. Default is \"0\".\n"
    "\n"
    "   --blend=replace     Disable blending (default).\n"
    "   --blend=srcover     Enable alpha blending.\n"
    "   --blend=depthonly   Render only to the depth buffer.\n"
    "\n"
    "   --log=<file.log>    Append benchmark results to a log file.\n"
    "   --image=<file.png>  Export resulting images.\n"
    "\n"
;

//------------------------------------------------------------------------
static const char* const s_guiHelpText =
    "General keys:\n"
    "\n"
    "   F1                 Hide this message\n"
    "   Esc, Alt-F4        Exit\n"
    "   <num>              Load state\n"
    "   Alt-<num>          Save state\n"
    "   F9                 Show/hide FPS counter\n"
    "   F10                Show/hide GUI\n"
    "   F11                Toggle fullscreen mode\n"
    "   PrtScn             Save screenshot\n"
    "\n"
    "Camera movement:\n"
    "\n"
    "   DragLeft, Arrows   Rotate\n"
    "   DragMiddle         Strafe\n"
    "   DragRight          Zoom\n"
    "   W, Alt-UpArrow     Move forward\n"
    "   S, Alt-DownArrow   Move back\n"
    "   A, Alt-LeftArrow   Strafe left\n"
    "   D, Alt-RightArrow  Strafe right\n"
    "   R, PageUp          Strafe up\n"
    "   F, PageDown        Strafe down\n"
    "   MouseWheel         Adjust movement speed\n"
    "   Space (hold)       Move faster\n"
    "   Ctrl (hold)        Move slower\n"
    "\n"
    "Uncheck \"Retain camera alignment\" to enable:\n"
    "\n"
    "   Q, Insert          Roll counter-clockwise\n"
    "   E, Home            Roll clockwise\n"
;

//------------------------------------------------------------------------

App::App(void)
:   m_commonCtrl            (CommonControls::Feature_Default & ~CommonControls::Feature_RepaintOnF5),
    m_cameraCtrl            (&m_commonCtrl, CameraControls::Feature_Default),

    m_action                (Action_None),
    m_showHelp              (false),
    m_showStats             (false),
    m_mode                  (Mode_CudaRaster),
    m_enableTexPhong        (true),
    m_enableDepth           (true),
    m_enableBlend           (false),
    m_enableLerp            (true),
    m_enableQuads           (false),
    m_numSamples            (1),

    m_mesh                  (NULL),
    m_textureAtlas          (ImageFormat::ABGR_8888),

    m_pipeDirty             (true),
    m_colorBuffer           (NULL),
    m_depthBuffer           (NULL),
    m_cudaModule            (NULL)
{
    // Initialize GUI.

    m_commonCtrl.showFPS(false);
    m_commonCtrl.addStateObject(this);
    m_commonCtrl.setStateFilePrefix("state_crsample_");

    m_commonCtrl.addToggle(&m_showHelp,                         FW_KEY_F1,      "Show help [F1]");
    m_commonCtrl.addButton((S32*)&m_action, Action_About,       FW_KEY_NONE,    "About...");
    m_commonCtrl.addButton((S32*)&m_action, Action_LoadMesh,    FW_KEY_M,       "Load mesh... [M]");
    m_commonCtrl.addToggle(&m_showStats,                        FW_KEY_F8,      "Show statistics [F8]");
    m_commonCtrl.addSeparator();

    m_commonCtrl.addToggle((S32*)&m_mode, Mode_CudaRaster,      FW_KEY_F2,      "CudaRaster [F2]");
    m_commonCtrl.addToggle((S32*)&m_mode, Mode_OpenGL,          FW_KEY_F3,      "OpenGL [F3]");
    m_commonCtrl.addSeparator();

    m_commonCtrl.addButton((S32*)&m_action, Action_ResetCamera, FW_KEY_NONE,    "Reset camera");
    m_commonCtrl.addButton((S32*)&m_action, Action_ExportCamera,FW_KEY_NONE,    "Export camera signature...");
    m_commonCtrl.addButton((S32*)&m_action, Action_ImportCamera,FW_KEY_NONE,    "Import camera signature...");
    m_window.addListener(&m_cameraCtrl);
    m_commonCtrl.addSeparator();

    m_commonCtrl.addToggle(&m_enableTexPhong,                   FW_KEY_G,       "Enable texturing and Phong shading [G]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_enableDepth,                      FW_KEY_H,       "Enable depth test [H]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_enableBlend,                      FW_KEY_J,       "Enable alpha blending [J]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_enableLerp,                       FW_KEY_K,       "CudaRaster: Enable attribute interpolation [K]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_enableQuads,                      FW_KEY_L,       "CudaRaster: Enable pixel-quad rendering (degrades perf) [L]", &m_pipeDirty);
    m_commonCtrl.addSeparator();

    m_commonCtrl.addToggle(&m_numSamples, 1,                    FW_KEY_Z,       "No MSAA [Z]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_numSamples, 2,                    FW_KEY_X,       "2x MSAA [X]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_numSamples, 4,                    FW_KEY_C,       "4x MSAA [C]", &m_pipeDirty);
    m_commonCtrl.addToggle(&m_numSamples, 8,                    FW_KEY_V,       "8x MSAA [V]", &m_pipeDirty);
    m_commonCtrl.addSeparator();

    m_window.setTitle("CudaRaster Sample Application");
    m_window.addListener(this);
    m_window.addListener(&m_commonCtrl);
    m_commonCtrl.flashButtonTitles();

    // Initialize CUDA compiler.

    m_cudaCompiler.setSourceFile("src/crsample/Shaders.cu");
    m_cudaCompiler.addOptions("-use_fast_math");
    m_cudaCompiler.include("src/framework");
    m_cudaCompiler.include("src/cudaraster");
    m_cudaCompiler.setMessageWindow(&m_window);

    // Load default state.

    if (!m_commonCtrl.loadState(m_commonCtrl.getStateFileName(1)))
        firstTimeInit();
    initPipe();
}

//------------------------------------------------------------------------

App::~App(void)
{
    setMesh(NULL);
    delete m_colorBuffer;
    delete m_depthBuffer;
}

//------------------------------------------------------------------------

bool App::handleEvent(const Window::Event& ev)
{
    // Window closed => exit.

    if (ev.type == Window::EventType_Close)
    {
        m_window.showModalMessage("Exiting...");
        delete this;
        return true;
    }

    // Handle actions.

    Action action = m_action;
    m_action = Action_None;
    String name;
    Mat4f mat;

    switch (action)
    {
    case Action_None:
        break;

    case Action_About:
        m_window.showMessageDialog("About", s_aboutText);
        break;

    case Action_LoadMesh:
        name = m_window.showFileLoadDialog("Load mesh", getMeshImportFilter(), "scenes/fairyforest");
        if (name.getLength())
            loadMesh(name);
        break;

    case Action_ResetCamera:
        if (m_mesh)
        {
            m_cameraCtrl.initForMesh(m_mesh);
            m_commonCtrl.message("Camera reset");
        }
        break;

    case Action_ExportCamera:
        m_window.setVisible(false);
        printf("\nCamera signature:\n");
        printf("%s\n", m_cameraCtrl.encodeSignature().getPtr());
        printf("Press any key to continue . . . ");
        _getch();
        printf("\n\n");
        break;

    case Action_ImportCamera:
        {
            m_window.setVisible(false);
            printf("\nEnter camera signature:\n");

            char buf[1024];
            if (fgets(buf, FW_ARRAY_SIZE(buf), stdin) != NULL)
                m_cameraCtrl.decodeSignature(buf);
            else
                setError("Signature too long!");

            if (!hasError())
                printf("Done.\n\n");
            else
            {
                printf("Error: %s\n", getError().getPtr());
                clearError();
                printf("Press any key to continue . . . ");
                _getch();
                printf("\n\n");
            }
        }
        break;

    default:
        FW_ASSERT(false);
        break;
    }

    // Set MSAA mode for OpenGL.

    if (m_mode == Mode_OpenGL)
    {
        GLContext::Config glConfig = m_window.getGLConfig();
        glConfig.numSamples = m_numSamples;
        m_window.setGLConfig(glConfig);
    }
    m_window.setVisible(true);

    // Render.

    if (ev.type == Window::EventType_Paint)
    {
        GLContext* gl = m_window.getGL();
        render(gl);

        if (m_commonCtrl.getShowControls())
        {
            gl->setVGXform(Mat4f());
            glDisable(GL_DEPTH_TEST);

            gl->setFont("Arial", 22, GLContext::FontStyle_Normal);
            gl->drawLabel((m_mode == Mode_CudaRaster) ? "CudaRaster" : "OpenGL", Vec2f(0.0f, 0.99f), Vec2f(0.5f, 1.0f), 0xFFFFFFFF);

            if (m_showHelp)
            {
                gl->setFont("Courier New", 18, GLContext::FontStyle_Bold);
                gl->drawLabel(s_guiHelpText, Vec2f(-0.99f, 0.99f), Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
            }

            gl->setDefaultFont();
        }
    }

    m_window.repaint();
    return false;
}

//------------------------------------------------------------------------

void App::readState(StateDump& d)
{
    d.pushOwner("App");
    String meshFileName;
    d.get(meshFileName,     "m_meshFileName");
    d.get(m_showStats,      "m_showStats");
    d.get((S32&)m_mode,     "m_mode");
    d.get(m_enableTexPhong, "m_enableTexPhong");
    d.get(m_enableDepth,    "m_enableDepth");
    d.get(m_enableBlend,    "m_enableBlend");
    d.get(m_enableLerp,     "m_enableLerp");
    d.get(m_enableQuads,    "m_enableQuads");
    d.get(m_numSamples,     "m_numSamples");
    d.popOwner();

    if (m_meshFileName != meshFileName && meshFileName.getLength())
        loadMesh(meshFileName);
    m_pipeDirty = true;
}

//------------------------------------------------------------------------

void App::writeState(StateDump& d) const
{
    d.pushOwner("App");
    d.set(m_meshFileName,   "m_meshFileName");
    d.set(m_showStats,      "m_showStats");
    d.set((S32)m_mode,      "m_mode");
    d.set(m_enableTexPhong, "m_enableTexPhong");
    d.set(m_enableDepth,    "m_enableDepth");
    d.set(m_enableBlend,    "m_enableBlend");
    d.set(m_enableLerp,     "m_enableLerp");
    d.set(m_enableQuads,    "m_enableQuads");
    d.set(m_numSamples,     "m_numSamples");
    d.popOwner();
}

//------------------------------------------------------------------------

void App::loadMesh(const String& fileName)
{
    m_window.showModalMessage(sprintf("Loading mesh from '%s'...\nThis will take a few seconds.", fileName.getFileName().getPtr()));
    String oldError = clearError();
    MeshBase* mesh = importMesh(fileName);
    String newError = getError();

    if (restoreError(oldError))
    {
        delete mesh;
        m_commonCtrl.message(sprintf("Error while loading '%s': %s", fileName.getPtr(), newError.getPtr()));
    }
    else
    {
        m_meshFileName = fileName;
        setMesh(mesh);
        m_commonCtrl.message(sprintf("Loaded mesh from '%s'", fileName.getPtr()));
    }
}

//------------------------------------------------------------------------

void App::setMesh(MeshBase* mesh)
{
    // Set mesh.

    delete m_mesh;
    m_mesh = mesh;
    if (!m_mesh)
        return;

    // Fix geometry.

    m_mesh->clean();                // Remove degenerates, etc.
    m_mesh->collapseVertices();     // Get rid of duplicate vertices.
    m_mesh->dupVertsPerSubmesh();   // Makes it possible to assign materials to vertices unambiguously.
    m_mesh->fixMaterialColors();    // Come up with reasonable material colors for Gouraud shading.

    // Convert to PNT.

    Mesh<VertexPNT> convMesh(*m_mesh);
    m_numVertices = convMesh.numVertices();
    m_numMaterials = convMesh.numSubmeshes();
    m_numTriangles = convMesh.numTriangles();

    // Allocate buffers.

    m_inputVertices.resizeDiscard       (m_numVertices  * sizeof(InputVertex));
    m_shadedVertices.resizeDiscard      (m_numVertices  * max(sizeof(ShadedVertex_gouraud), sizeof(ShadedVertex_texPhong)));
    m_materials.resizeDiscard           (m_numMaterials * sizeof(Material));
    m_vertexIndices.resizeDiscard       (m_numTriangles * sizeof(Vec3i));
    m_vertexMaterialIdx.resizeDiscard   (m_numVertices  * sizeof(S32));
    m_triangleMaterialIdx.resizeDiscard (m_numTriangles * sizeof(S32));

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

    // Construct texture atlas.

    m_textureAtlas.clear();
    for (int i = 0; i < m_numMaterials; i++)
        for (int j = MeshBase::TextureType_Diffuse; j <= MeshBase::TextureType_Alpha; j++)
            for (int k = 0; m_textureAtlas.addTexture(convMesh.material(i).textures[j].getMipLevel(k)); k++);

    // Setup materials.

    Material* materialPtr = (Material*)m_materials.getMutablePtr();
    for (int i = 0; i < m_numMaterials; i++)
    {
        const MeshBase::Material& in = convMesh.material(i);
        Material& out = materialPtr[i];
        out.diffuseColor    = in.diffuse;
        out.specularColor   = in.specular * 0.5f;
        out.glossiness      = in.glossiness;

        setupTexture(out.diffuseTexture, in.textures[MeshBase::TextureType_Diffuse]);
        setupTexture(out.alphaTexture, in.textures[MeshBase::TextureType_Alpha]);
    }

    // Setup material indices.

    S32* vertexMaterialIdxPtr = (S32*)m_vertexMaterialIdx.getMutablePtr();
    S32* triangleMaterialIdxPtr = (S32*)m_triangleMaterialIdx.getMutablePtr();
    for (int i = 0; i < m_numMaterials; i++)
    for (int j = 0; j < convMesh.indices(i).getSize(); j++)
    {
        for (int k = 0; k < 3; k++)
            vertexMaterialIdxPtr[convMesh.indices(i)[j][k]] = i;
        *triangleMaterialIdxPtr++ = i;
    }
}

//------------------------------------------------------------------------

void App::setupTexture(TextureSpec& spec, const Texture& tex)
{
    if (!tex.exists())
    {
        memset(&spec, 0, sizeof(TextureSpec));
        return;
    }

    TextureAtlas& atlas = m_textureAtlas;
    Vec4f coef = 1.0f / Vec4f(Vec4i(atlas.getAtlasSize(), atlas.getAtlasSize()));
    spec.size = Vec2f(tex.getSize());

    for (int i = 0; i < (int)FW_ARRAY_SIZE(spec.mipLevels); i++)
    {
        Texture mip = tex.getMipLevel(i);
        spec.mipLevels[i] = Vec4f(Vec4i(mip.getSize(), atlas.getTexturePos(mip))) * coef;
    }
}

//------------------------------------------------------------------------

void App::render(GLContext* gl)
{
    Mat4f posToCamera = m_cameraCtrl.getWorldToCamera();
    Mat4f projection = gl->xformFitToView(-1.0f, 2.0f) * m_cameraCtrl.getCameraToClip();

    m_commonCtrl.message("", "meshStats");
    m_commonCtrl.message("", "cudaRasterStats");

    // No mesh => skip.

    if (!m_mesh)
    {
        gl->drawModalMessage("No mesh loaded!");
        return;
    }

    // Show mesh statistics.

    if (m_showStats)
    {
        m_commonCtrl.message(sprintf("Mesh: triangles = %d, vertices = %d, materials = %d",
            m_numTriangles,
            m_numVertices,
            m_numMaterials
        ), "meshStats");
    }

    // Mode_OpenGL => special case.

    if (m_mode == Mode_OpenGL)
    {
        glClearColor(0.2f, 0.4f, 0.8f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        if (m_enableDepth)  glEnable(GL_DEPTH_TEST);
        if (m_enableBlend)  glEnable(GL_BLEND);

        m_mesh->draw(gl, posToCamera, projection, NULL, (!m_enableTexPhong));
        return;
    }

    // Parameters changed => reinitialize pipe.

    if (m_colorBuffer && m_colorBuffer->getSize() != m_window.getSize())
        m_pipeDirty = true;

    if (m_colorBuffer && m_colorBuffer->getNumSamples() != m_numSamples)
        m_pipeDirty = true;

    if (m_pipeDirty)
        initPipe();
    m_pipeDirty = false;

    // Set globals.

    Constants& c = *(Constants*)m_cudaModule->getGlobal("c_constants").getMutablePtrDiscard();
    c.posToClip             = projection * posToCamera;
    c.posToCamera           = posToCamera;
    c.normalToCamera        = transpose(invert(posToCamera.getXYZ()));
    c.materials             = m_materials.getCudaPtr();
    c.vertexMaterialIdx     = m_vertexMaterialIdx.getCudaPtr();
    c.triangleMaterialIdx   = m_triangleMaterialIdx.getCudaPtr();

    m_cudaModule->setTexRef("t_textureAtlas", m_textureAtlas.getAtlasTexture().getCudaArray());

    // Run vertex shader.

    m_vertexShaderKernel.setParams(m_inputVertices, m_shadedVertices, m_numVertices).launch(m_numVertices);

    // Run pixel pipe.

    m_cudaRaster.deferredClear(Vec4f(0.2f, 0.4f, 0.8f, 1.0f));

    m_cudaRaster.setVertexBuffer(&m_shadedVertices, 0);
    m_cudaRaster.setIndexBuffer(&m_vertexIndices, 0, m_numTriangles);
    m_cudaRaster.drawTriangles();

    m_colorBuffer->resolveToScreen(gl);

    // Show CudaRaster statistics.

    if (m_showStats)
    {
        CudaRaster::Stats s = m_cudaRaster.getStats();
        m_commonCtrl.message(sprintf("CudaRaster: setup = %.2fms, bin = %.2fms, coarse = %.2fms, fine = %.2fms, total = %.2fms",
            s.setupTime * 1.0e3f,
            s.binTime * 1.0e3f,
            s.coarseTime * 1.0e3f,
            s.fineTime * 1.0e3f,
            (s.setupTime + s.binTime + s.coarseTime + s.fineTime) * 1.0e3f
        ), "cudaRasterStats");
    }
}

//------------------------------------------------------------------------

void App::initPipe(void)
{
    const char* pipePostfix = (!m_enableTexPhong) ? "gouraud" : "texPhong";

    // Create surfaces.

    delete m_colorBuffer;
    delete m_depthBuffer;
    m_colorBuffer = new CudaSurface(m_window.getSize(), CudaSurface::Format_R8_G8_B8_A8, m_numSamples);
    m_depthBuffer = new CudaSurface(m_window.getSize(), CudaSurface::Format_Depth32, m_numSamples);

    // Compile CUDA code.

    U32 renderModeFlags = 0;
    if (m_enableDepth)  renderModeFlags |= RenderModeFlag_EnableDepth;
    if (m_enableLerp)   renderModeFlags |= RenderModeFlag_EnableLerp;
    if (m_enableQuads)  renderModeFlags |= RenderModeFlag_EnableQuads;

    m_cudaCompiler.clearDefines();
    m_cudaCompiler.define("SAMPLES_LOG2", m_colorBuffer->getSamplesLog2());
    m_cudaCompiler.define("RENDER_MODE_FLAGS", renderModeFlags);
    m_cudaCompiler.define("BLEND_SHADER", (!m_enableBlend) ? "BlendReplace" : "BlendSrcOver");

    m_cudaModule = m_cudaCompiler.compile();
    m_vertexShaderKernel = m_cudaModule->getKernel(String("vertexShader_") + pipePostfix);

    // Setup CudaRaster.

    m_cudaRaster.setSurfaces(m_colorBuffer, m_depthBuffer);
    m_cudaRaster.setPixelPipe(m_cudaModule, String("PixelPipe_") + pipePostfix);
}

//------------------------------------------------------------------------

void App::firstTimeInit(void)
{
    printf("Performing first-time initialization.\n");
    printf("This will take a while.\n");

    // Populate CudaCompiler cache.

    CudaCompiler::staticInit();

//  int numMSAA = 4, numModes = 8, numBlends = 2; // all variants
    int numMSAA = 1, numModes = 2, numBlends = 2; // first 3 toggles

    int progress = 0;
    for (int msaa = 0; msaa < numMSAA; msaa++)
    for (int mode = 0; mode < numModes; mode++)
    for (int blend = 0; blend < numBlends; blend++)
    {
        printf("\rPopulating CudaCompiler cache... %d/%d", ++progress, numMSAA * numModes * numBlends);
        m_cudaCompiler.clearDefines();
        m_cudaCompiler.define("SAMPLES_LOG2", msaa);
        m_cudaCompiler.define("RENDER_MODE_FLAGS", mode ^ RenderModeFlag_EnableDepth ^ RenderModeFlag_EnableLerp);
        m_cudaCompiler.define("BLEND_SHADER", (blend == 0) ? "BlendReplace" : "BlendSrcOver");
        m_cudaCompiler.compile(false);
    }
    printf("\rPopulating CudaCompiler cache... Done.\n");

    // Setup default state.

    printf("Loading mesh...\n");
    loadMesh("scenes/fairyforest/fairyforest.obj");
    m_cameraCtrl.decodeSignature("W4K3y/xWk1z/qei1z/5RI4py18I6Ew18/L67z////X105CKHv/Yta4000");
    m_commonCtrl.saveState(m_commonCtrl.getStateFileName(1));
    failIfError();

    // Print footer.

    printf("Done.\n");
    printf("\n");
}

//------------------------------------------------------------------------

void FW::init(void)
{
    // Parse mode.

    bool modeInteractive        = false;
    bool modeBenchSingle        = false;
    bool modeBenchResolution    = false;
    bool modeBenchRenderMode    = false;
    bool modeBenchProfile       = false;
    bool modeBenchSynthetic     = false;
    bool modeBenchBatch         = false;
    bool showHelp               = false;

    if (argc < 2)
    {
        printf("Specify \"--help\" for a list of command-line options.\n\n");
        modeInteractive = true;
    }
    else
    {
        String mode = argv[1];
        if (mode == "interactive")              modeInteractive = true;
        else if (mode == "bench-single")        modeBenchSingle = true;
        else if (mode == "bench-resolution")    modeBenchResolution = true;
        else if (mode == "bench-rendermode")    modeBenchRenderMode = true;
        else if (mode == "bench-profile")       modeBenchProfile = true;
        else if (mode == "bench-synthetic")     modeBenchSynthetic = true;
        else if (mode == "bench-batch")         modeBenchBatch = true;
        else                                    showHelp = true;
    }

    // Parse options.

    String meshFileName;
    Benchmark::Params p;

    for (int i = 2; i < argc; i++)
    {
        const char* ptr = argv[i];
        if ((parseLiteral(ptr, "--help") || parseLiteral(ptr, "-h")) && !*ptr)
        {
            showHelp = true;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--renderer=cuda") && !*ptr)
        {
            p.renderer = Benchmark::Renderer_CudaRaster;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--renderer=gl") && !*ptr)
        {
            p.renderer = Benchmark::Renderer_OpenGL;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--mesh="))
        {
            if (!*ptr)
                setError("Invalid mesh file '%s'!", argv[i]);
            meshFileName = ptr;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--camera="))
        {
            if (!*ptr)
                setError("Invalid camera signature '%s'!", argv[i]);
            p.cameras.add(ptr);
        }
        else if (!modeInteractive && parseLiteral(ptr, "--size="))
        {
            if (!parseInt(ptr, p.resolution.x) || !parseLiteral(ptr, "x") || !parseInt(ptr, p.resolution.y) || *ptr || min(p.resolution) <= 0)
                setError("Invalid resolution '%s'!", argv[i]);
        }
        else if (!modeInteractive && parseLiteral(ptr, "--samples="))
        {
            if (!parseInt(ptr, p.numSamples) || *ptr || p.numSamples < 1 || p.numSamples > 8 || popc8(p.numSamples) != 1)
                setError("Invalid number of samples '%s'!", argv[i]);
        }
        else if (!modeInteractive && parseLiteral(ptr, "--depth="))
        {
            if ((*ptr != '0' && *ptr != '1') || ptr[1])
                setError("Invalid --depth '%s'!", argv[i]);
            p.renderModeFlags &= ~RenderModeFlag_EnableDepth;
            if (*ptr == '1')
                p.renderModeFlags |= RenderModeFlag_EnableDepth;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--lerp="))
        {
            if ((*ptr != '0' && *ptr != '1') || ptr[1])
                setError("Invalid --lerp '%s'!", argv[i]);
            p.renderModeFlags &= ~RenderModeFlag_EnableLerp;
            if (*ptr == '1')
                p.renderModeFlags |= RenderModeFlag_EnableLerp;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--quads="))
        {
            if ((*ptr != '0' && *ptr != '1') || ptr[1])
                setError("Invalid --quads '%s'!", argv[i]);
            p.renderModeFlags &= ~RenderModeFlag_EnableQuads;
            if (*ptr == '1')
                p.renderModeFlags |= RenderModeFlag_EnableQuads;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--blend=replace") && !*ptr)
        {
            p.blend = Benchmark::Blend_Replace;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--blend=srcover") && !*ptr)
        {
            p.blend = Benchmark::Blend_SrcOver;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--blend=depthonly") && !*ptr)
        {
            p.blend = Benchmark::Blend_DepthOnly;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--log="))
        {
            if (!*ptr)
                setError("Invalid log file '%s'!", argv[i]);
            p.logFileName = ptr;
        }
        else if (!modeInteractive && parseLiteral(ptr, "--image="))
        {
            if (!*ptr)
                setError("Invalid image file '%s'!", argv[i]);
            p.imageFileName = ptr;
        }
        else
        {
            setError("Invalid option '%s'!", argv[i]);
        }
    }

    // Show help.

    if (showHelp)
    {
        printf("%s", s_commandHelpText);
        exitCode = 1;
        clearError();
        return;
    }

    // Validate options.

    if (modeBenchSingle || modeBenchResolution || modeBenchRenderMode || modeBenchProfile || modeBenchBatch)
    {
        if (!meshFileName.getLength())
            setError("Mesh file not specified!");
        if (!p.cameras.getSize())
            setError("No camera signature(s) specified!");
    }

    if (modeBenchProfile)
    {
        if (p.renderer != Benchmark::Renderer_CudaRaster)
            setError("Profiling supports only CudaRaster!");
        if (p.cameras.getSize() > 1)
            printf("Warning: Profiling supports only one camera!\n");
    }

    if (modeBenchSynthetic)
    {
        if (meshFileName.getLength())
            setError("Synthetic benchmark does not allow specifying a mesh!");
        if (p.cameras.getSize())
            setError("Synthetic benchmark does not allow specifying cameras!");
    }

    // Run.

    if (!hasError())
    {
        if (modeInteractive)
        {
            printf("Starting up...\n");
            App* app = new App;
            if (hasError())
                delete app;
        }
        else
        {
            Benchmark b;
            if (meshFileName.getLength())
                b.loadMesh(meshFileName);

            if (modeBenchSingle)        b.typeSingle(p);
            if (modeBenchResolution)    b.typeResolution(p);
            if (modeBenchRenderMode)    b.typeRenderMode(p);
            if (modeBenchProfile)       b.typeProfile(p);
            if (modeBenchSynthetic)     b.typeSynthetic(p);
            if (modeBenchBatch)         b.typeBatch(p);
        }
    }

    // Handle errors.

    if (hasError())
    {
        printf("Error: %s\n", getError().getPtr());
        exitCode = 1;
        clearError();
    }
}

//------------------------------------------------------------------------
