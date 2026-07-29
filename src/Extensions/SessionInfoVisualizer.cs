using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Numerics;
using System.Reactive;
using System.Reactive.Linq;
using System.Xml.Serialization;
using Bonsai;
using Hexa.NET.ImGui;

[Combinator]
public class SessionInfoVisualizer
{
    public bool Visible {get; set;}
    [XmlIgnore]
    public Dictionary<string, string> SessionInfo {get; set;}

    public IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    if (Visible)
                    {
                        var tableFlags = ImGuiTableFlags.Borders | ImGuiTableFlags.RowBg | ImGuiTableFlags.SizingStretchSame;

                        ImGui.Text("Session Info");
                        if (ImGui.BeginTable("Session Info", 2, tableFlags))
                        {
                            ImGui.TableSetupColumn("Field");
                            ImGui.TableSetupColumn("Value");

                            var r = 0;
                            foreach (var field in SessionInfo)
                            {
                                ImGui.TableNextRow();

                                ImGui.TableSetColumnIndex(0);
                                ImGui.Text(field.Key);

                                ImGui.TableSetColumnIndex(1);
                                ImGui.Text(field.Value);

                                r++;
                            }

                            ImGui.EndTable();
                        }
                        observer.OnNext(value);
                    }
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}

