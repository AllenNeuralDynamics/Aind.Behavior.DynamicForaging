using Hexa.NET.ImPlot;
using System;
using System.ComponentModel;
using Bonsai.ImPlot;

using ImPlot = Hexa.NET.ImPlot.ImPlot;
using Bonsai;
using System.Reactive.Linq;
using System.Reactive;
using System.Linq;
using System.Drawing;
using System.Numerics;

[Combinator]
public class LineStyling
{
    public float ColorR {get; set;}
    public float ColorG {get; set;}
    public float ColorB {get; set;}
    public float ColorA {get; set;}
    public float LineThickness {get; set;}

    public unsafe IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    ImPlot.SetNextLineStyle(new Vector4(ColorR, ColorG, ColorB, ColorA), LineThickness);
                    observer.OnNext(value);
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}