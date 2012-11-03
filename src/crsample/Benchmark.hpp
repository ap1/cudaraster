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

#pragma once
#include "3d/CameraControls.hpp"
#include "gpu/CudaCompiler.hpp"
#include "CudaRaster.hpp"

namespace FW
{
//------------------------------------------------------------------------

class Benchmark
{
public:
    enum Renderer
    {
        Renderer_CudaRaster,
        Renderer_OpenGL
    };

    enum Blend
    {
        Blend_Replace,
        Blend_SrcOver,
        Blend_DepthOnly,
    };

    struct Result
    {
        F32     totalTime;
        F32     setupTime;
        F32     binTime;
        F32     coarseTime;
        F32     fineTime;
        String  profilingInfo;

        Result(void)
        {
            totalTime = 0.0f, setupTime = 0.0f, binTime = 0.0f, coarseTime = 0.0f, fineTime = 0.0f;
        }

        Result& operator+=(const Result& r)
        {
            totalTime += r.totalTime, setupTime += r.setupTime, binTime += r.binTime, coarseTime += r.coarseTime, fineTime += r.fineTime;
            profilingInfo   += r.profilingInfo;
            return *this;
        }

        Result& operator*=(F32 c)
        {
            totalTime *= c, setupTime *= c, binTime *= c, coarseTime *= c, fineTime *= c;
            return *this;
        }
    };

    struct Params
    {
        Renderer        renderer;
        Vec2i           resolution;
        S32             numSamples;
        U32             renderModeFlags;
        Blend           blend;
        Array<String>   cameras;
        String          logFileName;
        String          imageFileName;

        Params(void)
        {
            renderer        = Renderer_CudaRaster;
            resolution      = Vec2i(1024, 768);
            numSamples      = 1;
            renderModeFlags = RenderModeFlag_EnableDepth | RenderModeFlag_EnableLerp;
            blend           = Blend_Replace;
        }
    };

public:
                        Benchmark       (void);
                        ~Benchmark      (void);

    // Utilities.

    void                log             (const String& fileName, const String& msg);
    String              appendFileName  (const String& fileName, const String& postfix);

    // Core functionality.

    void                loadMesh        (const String& fileName);
    void                initPipe        (const Vec2i& resolution, int numSamples, U32 renderModeFlags, Blend blend, int profilingMode);
    void                runVertexShader (const String& camera);
    Result              runSingle       (Renderer renderer, int numBatches, const String& imageFileName);
    Result              runCameras      (Renderer renderer, int numBatches, const String& imageFileName, const Array<String>& cameras);

    // Benchmark types.

    void                typeSingle      (const Params& p);
    void                typeResolution  (const Params& p); // Table 1
    void                typeRenderMode  (const Params& p); // Table 2
    void                typeProfile     (const Params& p); // Table 3
    void                typeSynthetic   (const Params& p); // Figure 5
    void                typeBatch       (const Params& p); // Figure 6

private:
                        Benchmark       (const Benchmark&); // forbidden
    Benchmark&          operator=       (const Benchmark&); // forbidden

private:
    Window              m_window;
    CameraControls      m_cameraCtrl;
    CudaCompiler        m_cudaCompiler;
    CudaRaster          m_cudaRaster;

    // Mesh.

    S32                 m_numVertices;
    S32                 m_numMaterials;
    S32                 m_numTriangles;
    Buffer              m_inputVertices;
    Buffer              m_shadedVertices;
    Buffer              m_materials;
    Buffer              m_vertexIndices;
    Buffer              m_vertexMaterialIdx;

    // Pipe.

    bool                m_enableLerp;
    CudaSurface*        m_colorBuffer;
    CudaSurface*        m_depthBuffer;
    CudaModule*         m_cudaModule;

    // GL framebuffer objects.

    GLuint              m_glDrawColorRenderbuffer;
    GLuint              m_glDrawDepthRenderbuffer;
    GLuint              m_glResolveColorRenderbuffer;
    GLuint              m_glDrawFramebuffer;
    GLuint              m_glResolveFramebuffer;
};

//------------------------------------------------------------------------
}
