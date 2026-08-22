using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Hexa.NET.ImGui;
using OpenTK.Graphics.OpenGL4;
using System.Runtime.Remoting.Contexts;


public class ShaderPass : Combinator<ImTextureRef, ImTextureRef>
{
    const string VertexShaderSource = @"
    #version 330 core
    layout(location = 0) in vec2 vertexPosition;
    layout(location = 1) in vec2 vertexTexCoords;

    out vec2 texCoord;

    void main()
    {
        texCoord = vec2(.5, .5) * (vertexPosition.xy + 1);
        gl_Position = vec4(vertexPosition.xy, 0.0, 1.0);
    }
    ";

    const string DefaultFragmentShader = @"
        #version 330 core
        uniform sampler2D tex0;
        uniform vec2 iResolution;
        in vec2 texCoord;
        out vec4 fragColor;

        void main()
        {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        ";

    static readonly float[] QuadVertices = new float[] { -1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f };

    public override IObservable<ImTextureRef> Process(IObservable<ImTextureRef> source)
    {
        return Observable.Defer(() =>
        {
            int shaderProgram = 0;
            int vertexArray = 0;
            int framebuffer = 0;
            int targetTexture = 0;
            int targetWidth = 0;
            int targetHeight = 0;
            var targetRef = default(ImTextureRef);
            return source.Select(texture =>
            {
                var currentContext = ImGui.GetCurrentContext();
                var sourceTexture = (int)(ulong)texture.GetTexID();
                if (sourceTexture == 0) return texture;

                if (shaderProgram == 0)
                {
                    shaderProgram = CreateProgram(VertexShaderSource, DefaultFragmentShader);
                    vertexArray = CreateQuad();
                    framebuffer = GL.GenFramebuffer();
                }

                // The pass renders into a texture of the same size as the incoming one.
                int width, height;
                GL.BindTexture(TextureTarget.Texture2D, sourceTexture);
                GL.GetTexLevelParameter(TextureTarget.Texture2D, 0, GetTextureParameter.TextureWidth, out width);
                GL.GetTexLevelParameter(TextureTarget.Texture2D, 0, GetTextureParameter.TextureHeight, out height);
                if (width != targetWidth || height != targetHeight)
                {
                    targetTexture = CreateTarget(targetTexture, width, height);
                    targetRef = CreateTextureRef(targetTexture);
                    targetWidth = width;
                    targetHeight = height;
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
                    GL.FramebufferTexture2D(
                        FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0,
                        TextureTarget.Texture2D, targetTexture, 0);
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                }

                var viewport = new int[4];
                GL.GetInteger(GetPName.Viewport, viewport);
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
                GL.Viewport(0, 0, width, height);
                GL.UseProgram(shaderProgram);

                GL.ActiveTexture(TextureUnit.Texture0);
                GL.BindTexture(TextureTarget.Texture2D, sourceTexture);

                GL.BindVertexArray(vertexArray);
                GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);

                GL.BindVertexArray(0);
                GL.UseProgram(0);
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                GL.Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);

                return targetRef;
            });
        });
    }

    static int CreateQuad()
    {
        var vertexArray = GL.GenVertexArray();
        GL.BindVertexArray(vertexArray);
        GL.BindBuffer(BufferTarget.ArrayBuffer, GL.GenBuffer());
        GL.BufferData(
            BufferTarget.ArrayBuffer, QuadVertices.Length * sizeof(float),
            QuadVertices, BufferUsageHint.StaticDraw);
        GL.EnableVertexAttribArray(0);
        GL.VertexAttribPointer(0, 2, VertexAttribPointerType.Float, false, 0, 0);
        GL.BindVertexArray(0);
        return vertexArray;
    }

    static int CreateTarget(int texture, int width, int height)
    {
        if (texture == 0) texture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, texture);
        GL.TexImage2D(
            TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba, width, height, 0,
            PixelFormat.Rgba, PixelType.UnsignedByte, IntPtr.Zero);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Linear);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Linear);
        return texture;
    }

    static ImTextureRef CreateTextureRef(int texture)
    {
        var textureRef = default(ImTextureRef);
        textureRef.TexID = new ImTextureID((IntPtr)texture);
        return textureRef;
    }

    static int CreateProgram(string vertexCode, string fragmentCode)
    {
        var vertexShader = CompileShader(ShaderType.VertexShader, vertexCode);
        var fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentCode);
        int status;

        var program = GL.CreateProgram();
        GL.AttachShader(program, vertexShader);
        GL.AttachShader(program, fragmentShader);
        GL.LinkProgram(program);
        GL.DetachShader(program, vertexShader);
        GL.DetachShader(program, fragmentShader);
        GL.GetProgram(program, GetProgramParameterName.LinkStatus, out status);

        if (status == 0)
        {
            var infoLog = GL.GetProgramInfoLog(program);
            GL.DeleteProgram(program);
            throw new InvalidOperationException(string.Format("Failed to link the shader program: {0}", infoLog));
        }

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        return program;
    }

    static int CompileShader(ShaderType type, string source)
    {
        int status;

        var shader = GL.CreateShader(type);
        GL.ShaderSource(shader, source);
        GL.CompileShader(shader);
        GL.GetShader(shader, ShaderParameter.CompileStatus, out status);

        if (status == 0)
        {
            var infoLog = GL.GetShaderInfoLog(shader);
            GL.DeleteShader(shader);
            throw new InvalidOperationException(string.Format("Failed to compile the {0}.", infoLog));
        }

        return shader;
    }
}
