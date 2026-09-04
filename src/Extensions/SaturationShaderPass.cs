using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Hexa.NET.ImGui;
using OpenTK.Graphics.OpenGL4;
using System.Runtime.Remoting.Contexts;


public class SaturationShaderPass : Combinator<ImTextureRef, ImTextureRef>
{
    public float Alpha { get; set; }
    public float MinSaturation {get; set;}
    public float MaxSaturation {get; set;}

    [Description("Clockwise rotation applied to the image, in radians.")]
    public float Rotation { get; set; }

    [Description("Mirrors the image left to right, about its vertical axis.")]
    public bool FlipHorizontal { get; set; }

    [Description("Mirrors the image top to bottom, about its horizontal axis.")]
    public bool FlipVertical { get; set; }

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

    const string SaturationFragmentShader = @"
        #version 330 core
        uniform sampler2D tex0;
        uniform vec2 iResolution;
        uniform float minSaturation = 0.1;
        uniform float maxSaturation = 0.9;
        uniform float alpha = 0.5;
        uniform float rotation = 0.0;
        uniform vec2 sourceSize;
        uniform vec2 outputSize;
        uniform vec2 flip = vec2(0.0, 0.0);
        in vec2 texCoord;
        out vec4 fragColor;

        // Maps a coordinate in the rotated output back into the source texture by
        // applying the inverse transform about the centre of the image. The rotation
        // offsets are in pixels rather than normalised units, so a non-square image is
        // rotated rigidly instead of being skewed by its aspect ratio. The forward
        // transform mirrors the image before rotating it, so the inverse undoes the
        // rotation first and the mirroring second. Each component of `flip` is 1 to
        // mirror that axis and 0 to leave it alone; mirroring maps the unit square onto
        // itself, so it does not disturb the coverage test in main().
        vec2 sourceCoord(vec2 uv)
        {
            vec2 offset = (uv - 0.5) * outputSize;
            float c = cos(rotation);
            float s = sin(rotation);
            vec2 rotated = vec2(c * offset.x + s * offset.y, c * offset.y - s * offset.x);
            vec2 sourceUv = rotated / sourceSize + 0.5;
            return mix(sourceUv, 1.0 - sourceUv, flip);
        }

        void main() 
        {
            vec2 sourceUv = sourceCoord(texCoord);

            // The rotated image does not cover the whole output; leave the rest clear.
            if (any(lessThan(sourceUv, vec2(0.0))) || any(greaterThan(sourceUv, vec2(1.0))))
            {
                fragColor = vec4(0,0,0,0);
                return;
            }

            vec4 texColor = texture(tex0, sourceUv);
            vec4 overlay = vec4(0,0,0,0);

            if (texColor.r >= maxSaturation) overlay = vec4(1,0,0,alpha);
            else if (texColor.r <= minSaturation) overlay = vec4(0,0,1,alpha);

            fragColor = mix(texColor, overlay, overlay.a);
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
            int alphaLocation = 0;
            int minSaturation = 0;
            int maxSaturation = 0;
            int rotationLocation = 0;
            int sourceSizeLocation = 0;
            int outputSizeLocation = 0;
            int flipLocation = 0;
            return source.Select(texture =>
            {
                var currentContext = ImGui.GetCurrentContext();
                var sourceTexture = (int)(ulong)texture.GetTexID();
                if (sourceTexture == 0) return texture;

                if (shaderProgram == 0)
                {
                    shaderProgram = CreateProgram(VertexShaderSource, SaturationFragmentShader);
                    vertexArray = CreateQuad();
                    framebuffer = GL.GenFramebuffer();
                    alphaLocation = GL.GetUniformLocation(shaderProgram, "alpha");
                    maxSaturation = GL.GetUniformLocation(shaderProgram, "maxSaturation");
                    minSaturation = GL.GetUniformLocation(shaderProgram, "minSaturation");
                    rotationLocation = GL.GetUniformLocation(shaderProgram, "rotation");
                    sourceSizeLocation = GL.GetUniformLocation(shaderProgram, "sourceSize");
                    outputSizeLocation = GL.GetUniformLocation(shaderProgram, "outputSize");
                    flipLocation = GL.GetUniformLocation(shaderProgram, "flip");
                }

                // The pass renders into a texture large enough to hold the bounding box
                // of the rotated image, so no part of the source is cropped away.
                int width, height;
                GL.BindTexture(TextureTarget.Texture2D, sourceTexture);
                GL.GetTexLevelParameter(TextureTarget.Texture2D, 0, GetTextureParameter.TextureWidth, out width);
                GL.GetTexLevelParameter(TextureTarget.Texture2D, 0, GetTextureParameter.TextureHeight, out height);

                var rotation = Rotation;
                var cos = Math.Abs(Math.Cos(rotation));
                var sin = Math.Abs(Math.Sin(rotation));
                var outputWidth = Math.Max(1, (int)Math.Round(width * cos + height * sin));
                var outputHeight = Math.Max(1, (int)Math.Round(width * sin + height * cos));
                if (outputWidth != targetWidth || outputHeight != targetHeight)
                {
                    targetTexture = CreateTarget(targetTexture, outputWidth, outputHeight);
                    targetRef = CreateTextureRef(targetTexture);
                    targetWidth = outputWidth;
                    targetHeight = outputHeight;
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
                    GL.FramebufferTexture2D(
                        FramebufferTarget.Framebuffer, FramebufferAttachment.ColorAttachment0,
                        TextureTarget.Texture2D, targetTexture, 0);
                    GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                }

                var viewport = new int[4];
                GL.GetInteger(GetPName.Viewport, viewport);
                var blendEnabled = GL.IsEnabled(EnableCap.Blend);
                GL.Disable(EnableCap.Blend);
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, framebuffer);
                GL.Viewport(0, 0, outputWidth, outputHeight);
                GL.UseProgram(shaderProgram);

                GL.ActiveTexture(TextureUnit.Texture0);
                GL.BindTexture(TextureTarget.Texture2D, sourceTexture);
                GL.Uniform1(alphaLocation, Alpha);
                GL.Uniform1(maxSaturation, MaxSaturation);
                GL.Uniform1(minSaturation, MinSaturation);
                GL.Uniform1(rotationLocation, rotation);
                GL.Uniform2(sourceSizeLocation, (float)width, (float)height);
                GL.Uniform2(outputSizeLocation, (float)outputWidth, (float)outputHeight);
                GL.Uniform2(flipLocation, FlipHorizontal ? 1f : 0f, FlipVertical ? 1f : 0f);

                GL.BindVertexArray(vertexArray);
                GL.DrawArrays(PrimitiveType.TriangleStrip, 0, 4);

                GL.BindVertexArray(0);
                GL.UseProgram(0);
                GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
                GL.Viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                if (blendEnabled) GL.Enable(EnableCap.Blend);

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
