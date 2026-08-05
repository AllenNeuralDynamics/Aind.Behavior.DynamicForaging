using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;

[Combinator]
[Description("Overlays a saturation warning on a grayscale image: pixels at or above MaxSaturation are tinted red (over-exposed), pixels at or below MinSaturation are tinted blue (under-exposed).")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class SaturationThreshold
{
    [Description("Intensity at or below which a pixel is marked blue (under-exposed).")]
    public double MinSaturation { get; set; }

    [Description("Intensity at or above which a pixel is marked red (over-exposed).")]
    public double MaxSaturation { get; set; }

    [Description("Blend strength of the overlay tint in the range [0, 1]. 1 = solid, 0 = invisible.")]
    public double Alpha { get; set; }

    public IObservable<IplImage> Process(IObservable<IplImage> source)
    {
        return source.Select(input =>
        {
            // The saturation test operates on a single intensity channel, matching the
            // fragment shader's use of texColor.r on a grayscale feed.
            var gray = input.Channels == 1 ? input : ExtractGray(input);

            // Build the two saturation masks (255 where the condition holds, else 0).
            var overMask = new IplImage(gray.Size, IplDepth.U8, 1);   // r >= MaxSaturation
            var underMask = new IplImage(gray.Size, IplDepth.U8, 1);  // r <= MinSaturation
            CV.Threshold(gray, overMask, MaxSaturation, 255, ThresholdTypes.Binary);
            CV.Threshold(gray, underMask, MinSaturation, 255, ThresholdTypes.BinaryInv);

            // The shader uses "else if", so red takes priority over blue.
            // Remove any overlap from the blue mask (a no-op when Min < Max).
            CV.Sub(underMask, overMask, underMask);

            // Colorize the grayscale image, then paint the overlays over it.
            var output = new IplImage(gray.Size, IplDepth.U8, 3);
            CV.CvtColor(gray, output, ColorConversion.Gray2Bgr);

            var alpha = Math.Max(0.0, Math.Min(1.0, Alpha));
            ApplyOverlay(output, overMask, new Scalar(0, 0, 255, 0), alpha);   // BGR red
            ApplyOverlay(output, underMask, new Scalar(255, 0, 0, 0), alpha);  // BGR blue
            return output;
        });
    }

    static IplImage ExtractGray(IplImage input)
    {
        var gray = new IplImage(input.Size, IplDepth.U8, 1);
        CV.CvtColor(input, gray, ColorConversion.Bgr2Gray);
        return gray;
    }

    // Reproduces mix(texColor, overlay, overlay.a): result = texColor * (1 - alpha) +
    // overlayColor * alpha, applied only on the masked pixels.
    static void ApplyOverlay(IplImage image, IplImage mask, Scalar color, double alpha)
    {
        if (alpha >= 1.0)
        {
            image.Set(color, mask); // fully opaque: just paint the tint
            return;
        }

        var solid = new IplImage(image.Size, image.Depth, image.Channels);
        solid.SetZero();
        solid.Set(color);

        var blended = new IplImage(image.Size, image.Depth, image.Channels);
        CV.AddWeighted(image, 1 - alpha, solid, alpha, 0, blended);
        CV.Copy(blended, image, mask); // write the blended result back only where the mask is set
    }
}
