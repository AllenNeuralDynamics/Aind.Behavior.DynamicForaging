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

    out vec2 TexCoords;

    void main()
    {
        TexCoords = vec2(.5, -.5) * (vertexPosition.xy + 1);
        gl_Position = vec4(vertexPosition.xy, 0.0, 1.0);
    }
    ";

    const string DefaultFragmentShader = @"
        #version 330
        uniform sampler2D tex0;
        uniform vec2 iResolution;
        in vec2 texCoord;
        out vec4 fragColor;

        void main()
        {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        }
        ";

    public override IObservable<ImTextureRef> Process(IObservable<ImTextureRef> source)
    {
        return Observable.Defer(() =>
        {
            int shaderProgram = 0;
            return source.Select(texture =>
            {
                var currentContext = ImGui.GetCurrentContext();
                var sourceTexture = (int)(ulong)texture.GetTexID();

                if (shaderProgram == 0)
                {
                    shaderProgram = CreateProgram(VertexShaderSource, DefaultFragmentShader);
                }

                // GL.UseProgram(shaderProgram);

                // GL.ActiveTexture(TextureUnit.Texture0);
                // GL.BindTexture(TextureTarget.Texture2D, sourceTexture);

                // GL.UseProgram(0);

                return texture;
            });
        });
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
            Console.WriteLine(infoLog);
            throw new InvalidOperationException("Failed to link the shader program.");
        }

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        return program;
    }

    static int CompileShader(ShaderType type, string source)
    {
        var shader = GL.CreateShader(type);
        return shader;
    }
}
