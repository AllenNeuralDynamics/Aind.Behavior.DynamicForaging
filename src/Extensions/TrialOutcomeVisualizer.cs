using Bonsai.Design;
using Bonsai.Expressions;
using Hexa.NET.ImGui;
using Hexa.NET.ImPlot;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Windows.Forms;
using AllenNeuralDynamics.Core.Design;
using AindDynamicForagingDataSchema;

public class TrialOutcomeVisualizer : BufferedVisualizer
{
    // Plot layout constants
    private const float MinPlotHeight = 100.0f;

    // Font size for text rendering
    private float fontSize = 16.0f;

    // Vertical bar constants (unrewarded is baseline, rewarded is 2x)
    private const float UnrewardedBarLength = 0.075f;
    private const float UnrewardedBarThickness = 2.0f;
    private const float RewardedBarLength = 0.15f;
    private const float RewardedBarThickness = 4.0f;

    private const double YAxisMin = -0.2;
    private const double YAxisMax = 1.2;

    // Y-axis positions for choices
    private const float RightChoiceY = 1.0f;
    private const float LeftChoiceY = 0.0f;
    private const float NoChoiceY = 0.5f;

    // Rolling average window size
    private const int RollingWindowSize = 10;

    // Colors (RGBA format)
    private static readonly Vector4 RightChoiceColor = new Vector4(0.0f, 0.6f, 0.0f, 1.0f);        // Green for right
    private static readonly Vector4 LeftChoiceColor = new Vector4(0.8f, 0.0f, 0.0f, 1.0f);         // Red for left
    private static readonly Vector4 NoChoiceColor = new Vector4(0.5f, 0.5f, 0.5f, 1.0f);           // Gray
    private static readonly Vector4 RightChoiceLineColor = new Vector4(0.0f, 0.6f, 0.0f, 1.0f);    // Green for right avg
    private static readonly Vector4 RewardRateLineColor = new Vector4(0.0f, 0.0f, 0.8f, 1.0f);     // Blue for reward rate
    private static readonly Vector4 ProbabilityTraceColor = new Vector4(0.0f, 0.0f, 0.0f, 1.0f);   // Black for probability traces

    // Line thickness for rolling averages
    private const float RollingLineThickness = 4.0f;

    // UI layout constants
    private const float InputWidth = 80.0f;

    ImGuiControl imGuiCanvas;

    private readonly List<TrialOutcome> trials = new List<TrialOutcome>();

    // History length (0 = show all trials)
    private int historyLength = 0;

    /// <inheritdoc/>
    public override void Show(object value)
    {
    }

    /// <inheritdoc/>
    protected override void ShowBuffer(IList<System.Reactive.Timestamped<object>> values)
    {
        imGuiCanvas.Invalidate();
        var casted = values.Select(v => (TrialOutcome)v.Value);
        foreach (var trial in casted)
        {
            trials.Add(trial);
        }
        base.ShowBuffer(values);
    }

    void StyleColors()
    {
        ImGui.StyleColorsLight();
        ImPlot.StyleColorsLight(ImPlot.GetStyle());
    }

    /// <summary>
    /// Computes rolling average over a window of values, ignoring null entries.
    /// </summary>
    private static float[] ComputeRollingAverage(float?[] values, int windowSize)
    {
        if (values == null || values.Length == 0)
        {
            return new float[0];
        }

        var result = new float[values.Length];

        for (int i = 0; i < values.Length; i++)
        {
            int windowStart = Math.Max(0, i - windowSize + 1);
            float sum = 0;
            int validCount = 0;

            for (int j = windowStart; j <= i; j++)
            {
                if (values[j].HasValue)
                {
                    sum += values[j].Value;
                    validCount++;
                }
            }

            result[i] = validCount > 0 ? sum / validCount : 0.5f;
        }

        return result;
    }

    /// <summary>
    /// Extracts choice values (1 for right, 0 for left, null for no choice) from trials.
    /// </summary>
    private static float?[] ExtractChoiceValues(List<TrialOutcome> trialList)
    {
        var values = new float?[trialList.Count];
        for (int i = 0; i < trialList.Count; i++)
        {
            var choice = trialList[i].IsRightChoice;
            if (choice.HasValue)
            {
                values[i] = choice.Value ? 1.0f : 0.0f;
            }
            else
            {
                values[i] = null;
            }
        }
        return values;
    }

    /// <summary>
    /// Extracts reward values (1 for rewarded, 0 for not rewarded) from trials.
    /// Only considers trials where a choice was made.
    /// </summary>
    private static float?[] ExtractRewardValues(List<TrialOutcome> trialList)
    {
        var values = new float?[trialList.Count];
        for (int i = 0; i < trialList.Count; i++)
        {
            if (trialList[i].IsRightChoice.HasValue)
            {
                values[i] = trialList[i].IsRewarded ? 1.0f : 0.0f;
            }
            else
            {
                values[i] = null;
            }
        }
        return values;
    }

    /// <summary>
    /// Extracts PRewardLeft values from trials.
    /// </summary>
    private static float[] ExtractPRewardLeft(List<TrialOutcome> trialList)
    {
        var values = new float[trialList.Count];
        for (int i = 0; i < trialList.Count; i++)
        {
            values[i] = (float)trialList[i].Trial.PRewardLeft;
        }
        return values;
    }

    /// <summary>
    /// Extracts PRewardRight values from trials.
    /// </summary>
    private static float[] ExtractPRewardRight(List<TrialOutcome> trialList)
    {
        var values = new float[trialList.Count];
        for (int i = 0; i < trialList.Count; i++)
        {
            values[i] = (float)trialList[i].Trial.PRewardRight;
        }
        return values;
    }

    /// <summary>
    /// Creates an array of trial indices for x-axis plotting.
    /// </summary>
    private static float[] CreateTrialIndices(int count)
    {
        var indices = new float[count];
        for (int i = 0; i < count; i++)
        {
            indices[i] = i;
        }
        return indices;
    }

    /// <summary>
    /// Plots vertical bars for trial outcomes.
    /// </summary>
    /// <param name="xValues">X positions of the bars</param>
    /// <param name="baseY">The base Y position (bottom of bar for upward, top for downward)</param>
    /// <param name="barLength">Length of the bar</param>
    /// <param name="barThickness">Thickness of the bar</param>
    /// <param name="direction">1 for upward (from baseY), -1 for downward (from baseY)</param>
    /// <param name="color">Color of the bars</param>
    /// <param name="label">Label for the legend</param>
    private static void PlotVerticalBars(
        List<float> xValues,
        float baseY,
        float barLength,
        float barThickness,
        int direction,
        Vector4 color,
        string label)
    {
        if (xValues.Count == 0) return;

        float endY = baseY + (barLength * direction);

        for (int i = 0; i < xValues.Count; i++)
        {
            float x = xValues[i];
            var xs = new float[] { x, x };
            var ys = new float[] { baseY, endY };

            // Set color before each line (SetNextLineStyle only applies to next plot call)
            ImPlot.SetNextLineStyle(color, barThickness);

            // Only use label for first bar to avoid duplicate legend entries
            string barLabel = (i == 0) ? label : "##" + label + i.ToString();
            ImPlot.PlotLine(barLabel, ref xs[0], ref ys[0], 2);
        }
    }

    /// <summary>
    /// Plots rewarded trial bars (double thickness and length).
    /// </summary>
    private static void PlotRewardedBars(
        List<float> xValues,
        float baseY,
        int direction,
        Vector4 color,
        string label)
    {
        PlotVerticalBars(xValues, baseY, RewardedBarLength, RewardedBarThickness, direction, color, label);
    }

    /// <summary>
    /// Plots unrewarded trial bars (baseline thickness and length).
    /// </summary>
    private static void PlotUnrewardedBars(
        List<float> xValues,
        float baseY,
        int direction,
        Vector4 color,
        string label)
    {
        PlotVerticalBars(xValues, baseY, UnrewardedBarLength, UnrewardedBarThickness, direction, color, label);
    }

    /// <summary>
    /// Plots no-choice markers (cross at middle).
    /// </summary>
    private static void PlotNoChoiceMarkers(
        List<float> xValues,
        Vector4 color,
        string label)
    {
        if (xValues.Count == 0) return;

        var yValues = new List<float>();
        for (int i = 0; i < xValues.Count; i++)
        {
            yValues.Add(NoChoiceY);
        }

        var xArray = xValues.ToArray();
        var yArray = yValues.ToArray();

        ImPlot.SetNextMarkerStyle(ImPlotMarker.Cross, 4.0f, color, 1.5f, color);
        ImPlot.PlotScatter(label, ref xArray[0], ref yArray[0], xArray.Length);
    }

    /// <summary>
    /// Plots a line for rolling averages.
    /// </summary>
    private static void PlotRollingLine(
        float[] xValues,
        float[] yValues,
        string label,
        Vector4 color)
    {
        if (xValues.Length == 0) return;

        ImPlot.SetNextLineStyle(color, RollingLineThickness);
        ImPlot.PlotLine(label, ref xValues[0], ref yValues[0], xValues.Length);
    }

    /// <summary>
    /// Categorizes trials and collects their x positions.
    /// </summary>
    private static void CategorizeTrials(
        List<TrialOutcome> trialList,
        out List<float> rightRewardedX,
        out List<float> rightUnrewardedX,
        out List<float> leftRewardedX,
        out List<float> leftUnrewardedX,
        out List<float> noChoiceX)
    {
        rightRewardedX = new List<float>();
        rightUnrewardedX = new List<float>();
        leftRewardedX = new List<float>();
        leftUnrewardedX = new List<float>();
        noChoiceX = new List<float>();

        for (int i = 0; i < trialList.Count; i++)
        {
            var trial = trialList[i];
            float x = i;

            if (!trial.IsRightChoice.HasValue)
            {
                noChoiceX.Add(x);
            }
            else if (trial.IsRightChoice.Value)
            {
                if (trial.IsRewarded)
                {
                    rightRewardedX.Add(x);
                }
                else
                {
                    rightUnrewardedX.Add(x);
                }
            }
            else
            {
                if (trial.IsRewarded)
                {
                    leftRewardedX.Add(x);
                }
                else
                {
                    leftUnrewardedX.Add(x);
                }
            }
        }
    }

    /// <summary>
    /// Calculates x-axis limits based on history length.
    /// </summary>
    private void GetXAxisLimits(int trialCount, out double xMin, out double xMax)
    {
        if (historyLength > 0 && trialCount > historyLength)
        {
            xMin = trialCount - historyLength - 0.5;
            xMax = trialCount;
        }
        else
        {
            xMin = -0.5;
            xMax = trialCount;
        }
    }

    void DrawTrials()
    {
        // Draw controls
        ImGui.Text("History:");
        ImGui.SameLine();
        ImGui.SetNextItemWidth(InputWidth);
        ImGui.InputInt("##history", ref historyLength);
        if (historyLength < 0) historyLength = 0;

        if (trials.Count == 0)
        {
            ImGui.Text("No trials to display.");
            return;
        }

        var availableSize = ImGui.GetContentRegionAvail();
        float plotHeight = Math.Max(availableSize.Y, MinPlotHeight);

        // Prepare data for rolling averages
        var trialIndices = CreateTrialIndices(trials.Count);
        var choiceValues = ExtractChoiceValues(trials);
        var rewardValues = ExtractRewardValues(trials);
        var rollingChoiceAvg = ComputeRollingAverage(choiceValues, RollingWindowSize);
        var rollingRewardRate = ComputeRollingAverage(rewardValues, RollingWindowSize);

        // Extract probability traces
        var pRewardLeft = ExtractPRewardLeft(trials);
        var pRewardRight = ExtractPRewardRight(trials);

        // Categorize trials
        List<float> rightRewardedX;
        List<float> rightUnrewardedX;
        List<float> leftRewardedX;
        List<float> leftUnrewardedX;
        List<float> noChoiceX;

        CategorizeTrials(trials,
            out rightRewardedX,
            out rightUnrewardedX,
            out leftRewardedX,
            out leftUnrewardedX,
            out noChoiceX);

        // Calculate x-axis limits
        double xMin, xMax;
        GetXAxisLimits(trials.Count, out xMin, out xMax);

        ImPlot.SetNextAxesLimits(xMin, xMax, YAxisMin, YAxisMax, ImPlotCond.Always);
        if (ImPlot.BeginPlot("Trial Outcomes", new Vector2(-1, plotHeight), ImPlotFlags.NoLegend | ImPlotFlags.NoTitle))
        {
            ImPlot.SetupAxes("Trial", "P(Right)");
            ImPlot.SetupAxisLimits(ImAxis.Y1, YAxisMin, YAxisMax, ImPlotCond.Always);

            // Plot probability traces
            PlotRollingLine(trialIndices, pRewardLeft, "P(Reward Left)", LeftChoiceColor);
            PlotRollingLine(trialIndices, pRewardRight, "P(Reward Right)", RightChoiceColor);

            PlotRollingLine(trialIndices, rollingChoiceAvg, "Right Choice Rate", ProbabilityTraceColor);
            PlotRollingLine(trialIndices, rollingRewardRate, "Reward Rate", RewardRateLineColor);

            // Right choices: bars extend upward from Y=1 (past 1)
            PlotRewardedBars(rightRewardedX, RightChoiceY, 1, RightChoiceColor, "Right + Rewarded");
            PlotUnrewardedBars(rightUnrewardedX, RightChoiceY, 1, RightChoiceColor, "Right + Unrewarded");

            // Left choices: bars extend downward from Y=0 (past 0 into negative)
            PlotRewardedBars(leftRewardedX, LeftChoiceY, -1, LeftChoiceColor, "Left + Rewarded");
            PlotUnrewardedBars(leftUnrewardedX, LeftChoiceY, -1, LeftChoiceColor, "Left + Unrewarded");

            // No choice: cross marker at middle
            PlotNoChoiceMarkers(noChoiceX, NoChoiceColor, "No Choice");

            ImPlot.EndPlot();
        }
    }

    /// <inheritdoc/>
    public override void Load(IServiceProvider provider)
    {
        var context = (ITypeVisualizerContext)provider.GetService(typeof(ITypeVisualizerContext));
        var visualizerBuilder = ExpressionBuilder.GetVisualizerElement(context.Source).Builder as TrialOutcomeVisualizerBuilder;
        if (visualizerBuilder != null)
        {
            fontSize = visualizerBuilder.FontSize;
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

            if (ImGui.Begin("TrialOutcomeVisualizer"))
            {
                DrawTrials();
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
                    ImGuiP.DockBuilderDockWindow("TrialOutcomeVisualizer", dockId);
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
        if (trials != null)
        {
            trials.Clear();
        }
    }
}

