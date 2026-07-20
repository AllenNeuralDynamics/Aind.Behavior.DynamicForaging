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
public class ToggleButtonBuilder : TextControlBuilder<bool>
{
    private bool Toggled;

    public string ToggledOffText {get; set;}
    public string ToggledOnText {get; set;}
    public bool Enabled {get; set;}

    /// <inheritdoc/>
    protected override IObservable<bool> Generate<TSource>(IObservable<TSource> source)
    {
        Toggled = false;
        return Observable.Create<bool>(observer =>
        {
            observer.OnNext(Toggled);
            var sourceObserver = Observer.Create<TSource>(
                _ =>
                {
                    if (Visible)
                    {
                        var label = Toggled ? ToggledOnText : ToggledOffText;
                        if (Enabled)
                        {
                            if (ImGui.Button(label))
                            {
                                observer.OnNext(!Toggled);
                                Toggled = !Toggled;
                            }
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