using AllenNeuralDynamics.AindBehaviorServices.DataTypes;
using Bonsai;
using Hexa.NET.ImGui;
using Hexa.NET.ImPlot;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using System.Reactive;
using Bonsai.Harp;
using System.Reactive.Linq;
using System.Collections.Immutable;
using System.Runtime.InteropServices;
using System.Drawing;
using System.Xml.Serialization;

[Combinator]
public class SoftwareEventVisualizer
{
    public bool Visible {get; set;}
    public ImmutableList<System.Reactive.Timestamped<SoftwareEvent>> SoftwareEvents {get; set;}
    public string TrialBreakEventName {get; set;}
    public float PlotHeight {get; set;}

    private const double YAxisMin = 0.0;
    private const double YAxisMax = 1.0;
    private const float InputWidth = 80.0f;

    private float timeWindow = 20.0f;
    public List<ShadedAreaPlotter> ShadedAreaPlotters {get; set;}
    public List<PointPlotter> PointPlotters {get; set;}
    private int maxTrials = 0;

    private readonly Dictionary<string, List<EventRecord>> eventHistory = new Dictionary<string, List<EventRecord>>();
    private readonly List<double> trialBreaks = new List<double>();
    private double lastTrialBreak = 0;
    private DateTimeOffset startTime;
    private double latestTimestamp = 0;

    private bool HasTrialBreaks { get { return !string.IsNullOrEmpty(TrialBreakEventName); } }
    private int TrialCount { get { return HasTrialBreaks ? trialBreaks.Count + 1 : 1; } }

    public IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            startTime = DateTimeOffset.Now;
            SoftwareEvents.Clear();
            eventHistory.Clear();
            trialBreaks.Clear();
            lastTrialBreak = 0;
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    if (Visible)
                    {
                        foreach (var v in SoftwareEvents)   
                        {
                            double timestamp = (v.Timestamp - startTime).TotalSeconds;

                            string name = v.Value.Name;
                            if (HasTrialBreaks && name == TrialBreakEventName)
                            {
                                if ((timestamp - lastTrialBreak) > 0.1)
                                {
                                    trialBreaks.Add(timestamp);
                                    lastTrialBreak = timestamp;
                                }
                            }

                            List<EventRecord> records;
                            if (!eventHistory.TryGetValue(name, out records))
                            {
                                records = new List<EventRecord>();
                                eventHistory[name] = records;
                            }
                            records.Add(new EventRecord { Timestamp = timestamp });
                        }   

                        CleanupOldEvents();

                        DrawEvents();

                        observer.OnNext(value);
                    }
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }

    private void CleanupOldEvents()
    {   
        latestTimestamp = (DateTimeOffset.Now - startTime).TotalSeconds;

        double cutoffTime = latestTimestamp - timeWindow;
        
        if (HasTrialBreaks && maxTrials > 0 && trialBreaks.Count > 0)
        {
            int firstVisible = Math.Max(0, TrialCount - maxTrials);
            if (firstVisible > 0 && firstVisible <= trialBreaks.Count)
            {
                double trialCutoff = trialBreaks[firstVisible - 1];
                cutoffTime = Math.Min(cutoffTime, trialCutoff);
            }
        }
        
        foreach (var kvp in eventHistory)
        {
            var records = kvp.Value;
            if (records.Count <= 1) continue;
            
            int keepFromIndex = -1;
            for (int i = records.Count - 1; i >= 0; i--)
            {
                if (records[i].Timestamp < cutoffTime)
                {
                    keepFromIndex = i;
                    break;
                }
            }
            
            if (keepFromIndex > 0)
            {
                records.RemoveRange(0, keepFromIndex);
            }
        }
        
        if (HasTrialBreaks && trialBreaks.Count > 1)
        {
            int keepFromIndex = -1;
            for (int i = trialBreaks.Count - 1; i >= 0; i--)
            {
                if (trialBreaks[i] < cutoffTime)
                {
                    keepFromIndex = i;
                    break;
                }
            }
            if (keepFromIndex > 0)
            {
                trialBreaks.RemoveRange(0, keepFromIndex);
            }
        }
    }

    private void DrawEvents()
    {
        latestTimestamp = (DateTimeOffset.Now - startTime).TotalSeconds;

        ImGui.Text("Time Window (s):");
        ImGui.SameLine();
        ImGui.SetNextItemWidth(InputWidth);
        ImGui.InputFloat("##timewindow", ref timeWindow);
        if (timeWindow < 1.0f) timeWindow = 1.0f;

        var availableSize = ImGui.GetContentRegionAvail();

        double plotTMin = -(double)timeWindow;
        double plotTMax = 0.0;

        int firstVisible, numVisible;
        GetVisibleTrialRange(out firstVisible, out numVisible);

        double yMin = HasTrialBreaks ? (double)firstVisible : YAxisMin;
        double yMax = HasTrialBreaks ? (double)(firstVisible + numVisible) : YAxisMax;

        ImPlot.SetNextAxesLimits(plotTMin, plotTMax, yMin, yMax, ImPlotCond.Always);
        if (ImPlot.BeginPlot("Software Events", new Vector2(-1, PlotHeight), ImPlotFlags.NoTitle))
        {
            ImPlot.SetupAxes("Time (s)", HasTrialBreaks ? "Trial" : "Value");
            ImPlot.SetupAxisLimits(ImAxis.Y1, yMin, yMax, ImPlotCond.Always);
            ImPlot.SetupLegend(ImPlotLocation.North, ImPlotLegendFlags.Outside | ImPlotLegendFlags.Horizontal);

            if (HasTrialBreaks && numVisible > 0)
            {
                SetupTrialAxisTicks(firstVisible, numVisible);
            }

            DrawAllShadedAreas(plotTMin, plotTMax);

            foreach (var config in PointPlotters)
            {
                DrawPointMarkers(config, plotTMin, plotTMax);
            }

            ImPlot.EndPlot();
        }
    }

    private void GetVisibleTrialRange(out int firstVisible, out int numVisible)
    {
        int total = TrialCount;
        if (!HasTrialBreaks)
        {
            firstVisible = 0;
            numVisible = 1;
        }
        else if (maxTrials > 0 && total > maxTrials)
        {
            firstVisible = total - maxTrials;
            numVisible = maxTrials;
        }
        else
        {
            firstVisible = 0;
            numVisible = total;
        }
    }

    unsafe private void SetupTrialAxisTicks(int firstVisibleTrial, int numVisibleTrials)
    {
        if (numVisibleTrials <= 0) return;

        var positions = new double[numVisibleTrials];
        var labelData = new byte[numVisibleTrials][];

        for (int t = 0; t < numVisibleTrials; t++)
        {
            int trialNum = firstVisibleTrial + t;
            positions[t] = trialNum + 0.5;
            labelData[t] = System.Text.Encoding.UTF8.GetBytes(trialNum.ToString() + '\0');
        }

        var handles = new GCHandle[numVisibleTrials];
        var ptrs = new IntPtr[numVisibleTrials];

        try
        {
            for (int t = 0; t < numVisibleTrials; t++)
            {
                handles[t] = GCHandle.Alloc(labelData[t], GCHandleType.Pinned);
                ptrs[t] = handles[t].AddrOfPinnedObject();
            }

            fixed (double* posPtr = positions)
            fixed (IntPtr* labelPtrs = ptrs)
            {
                ImPlot.SetupAxisTicks(ImAxis.Y1, posPtr, numVisibleTrials, (byte**)labelPtrs, false);
            }
        }
        finally
        {
            for (int t = 0; t < numVisibleTrials; t++)
            {
                if (handles[t].IsAllocated) handles[t].Free();
            }
        }
    }

    unsafe private void DrawAllShadedAreas(double plotTMin, double plotTMax)
    {
        if (ShadedAreaPlotters.Count == 0) return;

        var timeline = BuildMergedTimeline();
        if (timeline.Count == 0) return;

        double absMin = latestTimestamp + plotTMin;
        double absMax = latestTimestamp + plotTMax;

        int firstVisible, numVisible;
        GetVisibleTrialRange(out firstVisible, out numVisible);
        int lastVisible = firstVisible + numVisible;

        // Find the last event at or before the visible window start
        int startIdx = -1;
        for (int i = timeline.Count - 1; i >= 0; i--)
        {
            if (timeline[i].Timestamp <= absMin)
            {
                startIdx = i;
                break;
            }
        }

        if (startIdx < 0 && timeline.Count > 0 && timeline[0].Timestamp < absMax)
            startIdx = 0;
        if (startIdx < 0) return;

        for (int i = startIdx; i < timeline.Count; i++)
        {
            var segment = timeline[i];
            double segStart = Math.Max(segment.Timestamp, absMin);
            double segEnd = (i + 1 < timeline.Count) ? timeline[i + 1].Timestamp : absMax;

            if (segStart >= absMax) break;
            segEnd = Math.Min(segEnd, absMax);

            if (HasTrialBreaks)
            {
                int startTrial = GetTrialIndex(segStart);
                double currentStart = segStart;
                int currentTrial = startTrial;

                while (currentStart < segEnd)
                {
                    double trialEnd = (currentTrial < trialBreaks.Count)
                        ? trialBreaks[currentTrial]
                        : double.MaxValue;
                    double currentEnd = Math.Min(trialEnd, segEnd);

                    if (currentTrial >= firstVisible && currentTrial < lastVisible)
                    {
                        DrawShadedRect(segment.Config, currentStart, currentEnd, currentTrial);
                    }

                    currentStart = currentEnd;
                    currentTrial++;
                }
            }
            else
            {
                DrawShadedRect(segment.Config, segStart, segEnd, 0);
            }
        }
    }

    private double ToPlotTime(double timestamp)
    {
        return timestamp - latestTimestamp;
    }

    private static Vector4 ToVec4(Color color)
    {
        return new Vector4(color.R / 255f, color.G / 255f, color.B / 255f, color.A / 255f);
    }

    unsafe private void DrawShadedRect(ShadedAreaPlotter config, double tStart, double tEnd, int trialIndex)
    {
        double x0 = ToPlotTime(tStart);
        double x1 = ToPlotTime(tEnd);
        double yLow = HasTrialBreaks ? (double)trialIndex : 0.0;
        double yHigh = HasTrialBreaks ? (double)(trialIndex + 1) : 1.0;

        var color = ToVec4(config.Color);
        ImPlot.SetNextLineStyle(color, 0f);
        ImPlot.SetNextFillStyle(color, config.Alpha);

        fixed (double* xs = new double[] { x0, x1 })
        fixed (double* ysL = new double[] { yLow, yLow })
        fixed (double* ysH = new double[] { yHigh, yHigh })
        {
            ImPlot.PlotShaded(config.EventName, xs, ysL, ysH, 2);
        }
    }

    private int GetTrialIndex(double timestamp)
    {
        if (!HasTrialBreaks || trialBreaks.Count == 0) return 0;
        for (int i = trialBreaks.Count - 1; i >= 0; i--)
        {
            if (timestamp >= trialBreaks[i])
                return i + 1;
        }
        return 0;
    }

    unsafe private void DrawPointMarkers(PointPlotter config, double plotTMin, double plotTMax)
    {
        List<EventRecord> records;
        if (!eventHistory.TryGetValue(config.EventName, out records) || records.Count == 0)
            return;

        double absMin = latestTimestamp + plotTMin;
        double absMax = latestTimestamp + plotTMax;

        int firstVisible, numVisible;
        GetVisibleTrialRange(out firstVisible, out numVisible);
        int lastVisible = firstVisible + numVisible;

        var xsList = new List<double>();
        var ysList = new List<double>();

        for (int i = 0; i < records.Count; i++)
        {
            if (records[i].Timestamp < absMin) continue;
            if (records[i].Timestamp > absMax) continue;

            if (HasTrialBreaks)
            {
                int trial = GetTrialIndex(records[i].Timestamp);
                if (trial < firstVisible || trial >= lastVisible) continue;
                xsList.Add(ToPlotTime(records[i].Timestamp));
                ysList.Add((double)trial + (double)config.YPosition);
            }
            else
            {
                xsList.Add(ToPlotTime(records[i].Timestamp));
                ysList.Add((double)config.YPosition);
            }
        }

        if (xsList.Count == 0) return;

        var color = ToVec4(config.Color);
        ImPlot.SetNextMarkerStyle(config.Marker, config.MarkerSize, color, 1.5f, color);
        ImPlot.SetNextLineStyle(color, 0f);

        var xArr = xsList.ToArray();
        var yArr = ysList.ToArray();

        fixed (double* xs = xArr)
        fixed (double* ys = yArr)
        {
            ImPlot.PlotScatter(config.EventName, xs, ys, xArr.Length);
        }
    }

    private List<ShadedSegment> BuildMergedTimeline()
    {
        var merged = new List<ShadedSegment>();

        foreach (var config in ShadedAreaPlotters)
        {
            List<EventRecord> records;
            if (!eventHistory.TryGetValue(config.EventName, out records))
                continue;

            for (int i = 0; i < records.Count; i++)
            {
                merged.Add(new ShadedSegment
                {
                    Timestamp = records[i].Timestamp,
                    Config = config
                });
            }
        }

        merged.Sort(delegate(ShadedSegment a, ShadedSegment b)
        {
            return a.Timestamp.CompareTo(b.Timestamp);
        });

        return merged;
    }

    private struct EventRecord
    {
        public double Timestamp;
    }

    private struct ShadedSegment
    {
        public double Timestamp;
        public ShadedAreaPlotter Config;
    }

    public interface IPlotter
{
    string EventName { get; set; }
}

public class ShadedAreaPlotter : IPlotter
{
    private string _eventName;
    private Color _color;
    private float _alpha;

    public ShadedAreaPlotter()
    {
        _eventName = "";
        _color = Color.CornflowerBlue;
        _alpha = 0.3f;
    }

    [Description("The software event name to filter on.")]
    public string EventName
    {
        get { return _eventName; }
        set { _eventName = value; }
    }

    [XmlIgnore]
    [Description("The color of the shaded area.")]
    public Color Color
    {
        get { return _color; }
        set { _color = value; }
    }

    [Browsable(false)]
    [XmlElement("Color")]
    public string ColorHtml
    {
        get { return ColorTranslator.ToHtml(Color); }
        set { try { Color = ColorTranslator.FromHtml(value); } catch { } }
    }

    [Description("The transparency of the shaded area (0.0 to 1.0).")]
    public float Alpha
    {
        get { return _alpha; }
        set { _alpha = value; }
    }
}

public class PointPlotter : IPlotter
{
    private string _eventName;
    private Color _color;
    private float _yPosition;
    private float _markerSize;
    private ImPlotMarker _marker;

    public PointPlotter()
    {
        _eventName = "";
        _color = Color.Red;
        _yPosition = 0.5f;
        _markerSize = 6.0f;
        _marker = ImPlotMarker.Circle;
    }

    [Description("The software event name to filter on.")]
    public string EventName
    {
        get { return _eventName; }
        set { _eventName = value; }
    }

    [XmlIgnore]
    [Description("The color of the marker.")]
    public Color Color
    {
        get { return _color; }
        set { _color = value; }
    }

    [Browsable(false)]
    [XmlElement("Color")]
    public string ColorHtml
    {
        get { return ColorTranslator.ToHtml(Color); }
        set { try { Color = ColorTranslator.FromHtml(value); } catch { } }
    }

    [Description("The fixed Y position of the marker (0.0 to 1.0).")]
    public float YPosition
    {
        get { return _yPosition; }
        set { _yPosition = value; }
    }

    [Description("The size of the marker in pixels.")]
    public float MarkerSize
    {
        get { return _markerSize; }
        set { _markerSize = value; }
    }

    [Description("The marker style.")]
    public ImPlotMarker Marker
    {
        get { return _marker; }
        set { _marker = value; }
    }
}
}