using Bonsai.Design;
using Bonsai.Expressions;
using Bonsai.Harp;
using AllenNeuralDynamics.AindBehaviorServices.DataTypes;
using AllenNeuralDynamics.Core.Design;
using Hexa.NET.ImGui;
using Hexa.NET.ImPlot;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Numerics;
using System.Windows.Forms;

public class SoftwareEventVisualizer : BufferedVisualizer
{
    private const float MinPlotHeight = 100.0f;
    private const double YAxisMin = 0.0;
    private const double YAxisMax = 1.0;
    private const float InputWidth = 80.0f;

    private float fontSize = 16.0f;
    private float timeWindow = 30.0f;

    private List<ShadedAreaPlotter> shadedAreaPlotters = new List<ShadedAreaPlotter>();
    private List<PointPlotter> pointPlotters = new List<PointPlotter>();

    private ImGuiControl imGuiCanvas;

    private readonly Dictionary<string, List<EventRecord>> eventHistory = new Dictionary<string, List<EventRecord>>();
    private double latestTimestamp = 0;

    private struct EventRecord
    {
        public double Timestamp;
        public double Value;
    }

    private struct ShadedSegment
    {
        public double Timestamp;
        public ShadedAreaPlotter Config;
    }

    /// <inheritdoc/>
    public override void Show(object value)
    {
    }

    /// <inheritdoc/>
    protected override void ShowBuffer(IList<System.Reactive.Timestamped<object>> values)
    {
        imGuiCanvas.Invalidate();
        foreach (var v in values)
        {
            // Clock tick: just update the latest timestamp for plot scrolling
            if (v.Value is double)
            {
                var clockTimestamp = (double)v.Value;
                if (clockTimestamp > latestTimestamp)
                    latestTimestamp = clockTimestamp;
                continue;
            }

            SoftwareEvent softwareEvent;
            double timestamp;

            if (v.Value is Timestamped<SoftwareEvent>)
            {
                var harpTimestamped = (Timestamped<SoftwareEvent>)v.Value;
                softwareEvent = harpTimestamped.Value;
                timestamp = harpTimestamped.Seconds;
            }
            else if (v.Value is SoftwareEvent)
            {
                softwareEvent = (SoftwareEvent)v.Value;
                timestamp = softwareEvent.Timestamp ?? v.Timestamp.Ticks / (double)TimeSpan.TicksPerSecond;
            }
            else
            {
                continue;
            }

            if (timestamp > latestTimestamp)
                latestTimestamp = timestamp;

            string name = softwareEvent.Name;
            if (string.IsNullOrEmpty(name)) continue;

            double value = 0.5;
            if (softwareEvent.Data != null)
            {
                try { value = Convert.ToDouble(softwareEvent.Data); }
                catch { value = 0.5; }
            }

            List<EventRecord> records;
            if (!eventHistory.TryGetValue(name, out records))
            {
                records = new List<EventRecord>();
                eventHistory[name] = records;
            }
            records.Add(new EventRecord { Timestamp = timestamp, Value = value });
        }
        base.ShowBuffer(values);
    }

    private static Vector4 ToVec4(Color color)
    {
        return new Vector4(color.R / 255f, color.G / 255f, color.B / 255f, color.A / 255f);
    }

    void StyleColors()
    {
        ImGui.StyleColorsLight();
        ImPlot.StyleColorsLight(ImPlot.GetStyle());
    }

    /// <summary>
    /// Converts an absolute timestamp to plot-relative time where 0 = now.
    /// </summary>
    private double ToPlotTime(double timestamp)
    {
        return timestamp - latestTimestamp;
    }

    /// <summary>
    /// Builds a merged timeline of all shaded area events, sorted by timestamp.
    /// </summary>
    private List<ShadedSegment> BuildMergedTimeline()
    {
        var merged = new List<ShadedSegment>();

        foreach (var config in shadedAreaPlotters)
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

    /// <summary>
    /// Draws shaded areas as mutually exclusive colored regions spanning full Y (0-1).
    /// Each segment is drawn as a PlotShaded bar using fixed pointers (proven pattern).
    /// </summary>
    unsafe private void DrawAllShadedAreas(double plotTMin, double plotTMax)
    {
        if (shadedAreaPlotters.Count == 0) return;

        var timeline = BuildMergedTimeline();
        if (timeline.Count == 0) return;

        double absMin = latestTimestamp + plotTMin;
        double absMax = latestTimestamp + plotTMax;

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

            double x0 = ToPlotTime(segStart);
            double x1 = ToPlotTime(segEnd);

            var color = ToVec4(segment.Config.Color);

            ImPlot.SetNextLineStyle(color, 0f);
            ImPlot.SetNextFillStyle(color, segment.Config.Alpha);

            string label = "##shaded_" + i.ToString();

            fixed (double* xs = new double[] { x0, x1 })
            fixed (double* ysLow = new double[] { 0.0, 0.0 })
            fixed (double* ysHigh = new double[] { 1.0, 1.0 })
            {
                ImPlot.PlotShaded(label, xs, ysLow, ysHigh, 2);
            }
        }
    }

    /// <summary>
    /// Draws point markers using fixed pointers (proven pattern from PatchStateVisualizer).
    /// </summary>
    unsafe private void DrawPointMarkers(PointPlotter config, double plotTMin, double plotTMax)
    {
        List<EventRecord> records;
        if (!eventHistory.TryGetValue(config.EventName, out records) || records.Count == 0)
            return;

        double absMin = latestTimestamp + plotTMin;
        double absMax = latestTimestamp + plotTMax;

        var xsList = new List<double>();
        var ysList = new List<double>();

        for (int i = 0; i < records.Count; i++)
        {
            if (records[i].Timestamp < absMin) continue;
            if (records[i].Timestamp > absMax) continue;
            xsList.Add(ToPlotTime(records[i].Timestamp));
            ysList.Add((double)config.YPosition);
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

    unsafe private void DrawEvents()
    {
        ImGui.Text("Time Window (s):");
        ImGui.SameLine();
        ImGui.SetNextItemWidth(InputWidth);
        ImGui.InputFloat("##timewindow", ref timeWindow);
        if (timeWindow < 1.0f) timeWindow = 1.0f;

        if (latestTimestamp == 0)
        {
            ImGui.Text("No events received.");
            return;
        }

        var availableSize = ImGui.GetContentRegionAvail();
        float plotHeight = Math.Max(availableSize.Y, MinPlotHeight);

        // X axis: 0 = now (right), -timeWindow = oldest visible (left)
        double plotTMin = -(double)timeWindow;
        double plotTMax = 0.0;

        ImPlot.SetNextAxesLimits(plotTMin, plotTMax, YAxisMin, YAxisMax, ImPlotCond.Always);
        if (ImPlot.BeginPlot("Software Events", new Vector2(-1, plotHeight), ImPlotFlags.NoLegend | ImPlotFlags.NoTitle))
        {
            ImPlot.SetupAxes("Time (s)", "Value");
            ImPlot.SetupAxisLimits(ImAxis.Y1, YAxisMin, YAxisMax, ImPlotCond.Always);

            // Draw shaded areas (mutually exclusive timeline)
            DrawAllShadedAreas(plotTMin, plotTMax);

            // Draw point markers on top
            foreach (var config in pointPlotters)
            {
                DrawPointMarkers(config, plotTMin, plotTMax);
            }

            ImPlot.EndPlot();
        }
    }

    /// <inheritdoc/>
    public override void Load(IServiceProvider provider)
    {
        var context = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
        var builder = ExpressionBuilder.GetVisualizerElement(context.Source).Builder as SoftwareEventVisualizerBuilder;
        if (builder != null)
        {
            fontSize = builder.FontSize;
            timeWindow = builder.TimeWindow;
            shadedAreaPlotters = builder.ShadedAreaPlotters ?? new List<ShadedAreaPlotter>();
            pointPlotters = builder.PointPlotters ?? new List<PointPlotter>();
        }

        imGuiCanvas = new ImGuiControl();
        imGuiCanvas.Dock = DockStyle.Fill;
        imGuiCanvas.Render += (sender, e) =>
        {
            var dockspaceId = ImGui.DockSpaceOverViewport(
                0,
                ImGui.GetMainViewport(),
                ImGuiDockNodeFlags.AutoHideTabBar | ImGuiDockNodeFlags.NoUndocking);

            StyleColors();
            ImGui.PushFont(ImGui.GetFont(), fontSize);

            if (ImGui.Begin("SoftwareEventVisualizer"))
            {
                DrawEvents();
            }

            ImGui.End();
            ImGui.PopFont();
            var centralNode = ImGuiP.DockBuilderGetCentralNode(dockspaceId);
            if (!ImGui.IsWindowDocked() && !centralNode.IsNull)
            {
                unsafe
                {
                    var handle = centralNode.Handle;
                    uint dockId = handle->ID;
                    ImGuiP.DockBuilderDockWindow("SoftwareEventVisualizer", dockId);
                }
            }
        };

        var visualizerService = (IDialogTypeVisualizerService)provider.GetService(typeof(IDialogTypeVisualizerService));
        if (visualizerService != null)
        {
            visualizerService.AddControl(imGuiCanvas);
        }
    }

    /// <inheritdoc/>
    public override void Unload()
    {
        if (imGuiCanvas != null)
        {
            imGuiCanvas.Dispose();
        }
        eventHistory.Clear();
    }
}
