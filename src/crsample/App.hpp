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
#include "Shaders.hpp"
#include "gui/CommonControls.hpp"
#include "3d/CameraControls.hpp"
#include "3d/TextureAtlas.hpp"
#include "gpu/CudaCompiler.hpp"
#include "CudaRaster.hpp"

namespace FW
{
//------------------------------------------------------------------------

class App : public Window::Listener, public CommonControls::StateObject
{
private:
    enum Action
    {
        Action_None,

        Action_About,
        Action_LoadMesh,

        Action_ResetCamera,
        Action_ExportCamera,
        Action_ImportCamera,
    };

    enum Mode
    {
        Mode_CudaRaster = 0,
        Mode_OpenGL,
    };

public:
                        App             (void);
    virtual             ~App            (void);

    virtual bool        handleEvent     (const Window::Event& ev);
    virtual void        readState       (StateDump& d);
    virtual void        writeState      (StateDump& d) const;

private:
    void                loadMesh        (const String& fileName);
    void                setMesh         (MeshBase* mesh);
    void                setupTexture    (TextureSpec& spec, const Texture& tex);

    void                render          (GLContext* gl);
    void                initPipe        (void);

    void                firstTimeInit   (void);

private:
                        App             (const App&); // forbidden
    App&                operator=       (const App&); // forbidden

private:
    Window              m_window;
    CommonControls      m_commonCtrl;
    CameraControls      m_cameraCtrl;
    CudaCompiler        m_cudaCompiler;
    CudaRaster          m_cudaRaster;

    // State.

    Action              m_action;
    String              m_meshFileName;
    bool                m_showHelp;
    bool                m_showStats;
    Mode                m_mode;
    bool                m_enableTexPhong;
    bool                m_enableDepth;
    bool                m_enableBlend;
    bool                m_enableLerp;
    bool                m_enableQuads;
    S32                 m_numSamples;

    // Mesh.

    MeshBase*           m_mesh;
    S32                 m_numVertices;
    S32                 m_numMaterials;
    S32                 m_numTriangles;
    Buffer              m_inputVertices;
    Buffer              m_shadedVertices;
    Buffer              m_materials;
    Buffer              m_vertexIndices;
    Buffer              m_vertexMaterialIdx;
    Buffer              m_triangleMaterialIdx;
    TextureAtlas        m_textureAtlas;

    // Pipe.

    bool                m_pipeDirty;
    CudaSurface*        m_colorBuffer;
    CudaSurface*        m_depthBuffer;
    CudaModule*         m_cudaModule;
    CudaKernel          m_vertexShaderKernel;
};

//------------------------------------------------------------------------
}
