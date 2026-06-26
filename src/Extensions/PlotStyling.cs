using Hexa.NET.ImPlot;
using System;
using System.ComponentModel;
using Bonsai.ImPlot;

using ImPlot = Hexa.NET.ImPlot.ImPlot;
using Bonsai;
using System.Reactive.Linq;
using System.Reactive;
using System.Linq;

[Combinator]
public class PlotStyling
{
    public double[] TickPositions {get; set;}
    public string[] TickLabels {get; set;}
    public ImPlotLocation ImPlotLocationFlags {get; set;}
    public ImPlotLegendFlags ImPlotLegendFlags {get; set;}

    public unsafe IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    if (TickPositions.Count() > 0)
                    {
                        if (TickLabels.Count() == TickPositions.Count())
                        {
                            ImPlot.SetupAxisTicks(ImAxis.Y1, ref TickPositions[0], TickPositions.Count(), TickLabels);
                        } else
                        {
                            ImPlot.SetupAxisTicks(ImAxis.Y1, ref TickPositions[0], TickPositions.Count());
                        }
                    }
                    ImPlot.SetupLegend(ImPlotLocationFlags, ImPlotLegendFlags);
                    observer.OnNext(value);
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }
}

