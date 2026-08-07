using Hexa.NET.ImPlot;
using System;
using ImPlot = Hexa.NET.ImPlot.ImPlot;
using Bonsai;
using System.Reactive.Linq;
using System.Reactive;
using System.Linq;
using System.Numerics;

[Combinator]
public class ScatterStyling
{
    public ImPlotMarker ImPlotMarker {get; set;}
    public float Size {get; set;}
    public float FillColorR {get; set;}
    public float FillColorG {get; set;}
    public float FillColorB {get; set;}
    public float FillColorA {get; set;}
    public float OutlineColorR {get; set;}
    public float OutlineColorG {get; set;}
    public float OutlineColorB {get; set;}
    public float OutlineColorA {get; set;}
    public float OutlineWeight {get; set;}

    public unsafe IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    ImPlot.SetNextMarkerStyle(ImPlotMarker, Size, new Vector4(FillColorR, FillColorG, FillColorB, FillColorA), OutlineWeight, new Vector4(OutlineColorR, OutlineColorG, OutlineColorB, OutlineColorA));
                    observer.OnNext(value);
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}
