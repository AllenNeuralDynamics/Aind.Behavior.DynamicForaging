using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Linq;
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

    private Dictionary<string, string> _sessionInfo = new Dictionary<string, string>();

    [XmlIgnore]
    public object SessionInfo
    {
        get { return _sessionInfo; }
        set
        {
            _sessionInfo = ObjectToStringDict(value);
        }
    }

    public SessionInfoVisualizer()
    {
    }

    private static Dictionary<string, string> ObjectToStringDict(object obj)
    {
        if (obj == null) return new Dictionary<string, string>();
        var result = new Dictionary<string, string>();
        var jo = (Newtonsoft.Json.Linq.JObject)obj;
        foreach (var prop in jo.Properties())
        {
            result[prop.Name] = prop.Value != null ? prop.Value.ToString() : string.Empty;
        }
        return result;
    }

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

                                KeyValuePair<string, string>[] fields;
                                fields = _sessionInfo.ToArray();
                                foreach (var field in fields)
                                {
                                    ImGui.TableNextRow();

                                    ImGui.TableSetColumnIndex(0);
                                    ImGui.Text(field.Key);

                                    ImGui.TableSetColumnIndex(1);
                                    ImGui.Text(field.Value);
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

