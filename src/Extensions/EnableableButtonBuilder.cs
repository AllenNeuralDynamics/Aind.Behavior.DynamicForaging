using System;
using System.ComponentModel;
using System.Reactive;
using System.Reactive.Linq;
using Bonsai.ImGui;
using ImGui = Hexa.NET.ImGui.ImGui;

/// <summary>
/// Represents an operator that draws a button control and generates
/// a sequence of notifications whenever the button is clicked.
/// </summary>
[Description("Draws a button control and generates a sequence of notifications whenever the button is clicked.")]
public class EnableableButtonBuilder : TextControlBuilder<string>
{
    public bool Enabled {get; set;}

    /// <inheritdoc/>
    protected override IObservable<string> Generate<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<string>(observer =>
        {
            var capturedName = Name;
            var capturedText = Text;
            var idSuffix = "##" + capturedName;
            var label = capturedText + idSuffix;
            var output = capturedName ?? capturedText;
            var sourceObserver = Observer.Create<TSource>(
                _ =>
                {
                    var text = Text;
                    if (text != capturedText)
                    {
                        capturedText = text;
                        label = text + idSuffix;
                        output = capturedName ?? text;
                    }
                    if (Visible)
                    {
                        if (Enabled)
                        {
                            if (ImGui.Button(label))
                                observer.OnNext(output);
                        } else
                        {
                            ImGui.BeginDisabled();
                            ImGui.Button(label);
                            ImGui.EndDisabled();
                        }
                    }
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}