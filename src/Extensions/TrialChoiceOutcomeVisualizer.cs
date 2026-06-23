using AindDynamicForagingDataSchema;
using Bonsai;
using Hexa.NET.ImPlot;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Numerics;
using System.Reactive;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("Draws a progress bar showing a value between 0 and 1.")]
public class TrialChoiceOutcomeVisualizer
{
    public bool Visible {get; set;}
    public List<TrialOutcome> TrialOutcomes {get; set;}
    public float MarkerSize {get; set;}
    public float MarkerWeight {get; set;}
    public float TopPlotHeight {get; set;}
    public float BottomPlotHeight {get; set;}
    public float RollingLineThickness {get; set;}
    public int RollingWindowSize {get; set;}

    public IObservable<TSource> Process<TSource>(IObservable<TSource> source)
    {
        return Observable.Create<TSource>(observer =>
        {
            var sourceObserver = Observer.Create<TSource>(
                value =>
                {
                    var trialIndices = CreateTrialIndices(TrialOutcomes.Count);
                    var choiceValues = ExtractChoiceValues(TrialOutcomes);
                    var rewardValues = ExtractRewardValues(TrialOutcomes);
                    var finishedTrialValues = ExtractFinishedTrialValues(TrialOutcomes);
                    var rollingChoiceAvg = ComputeRollingAverage(choiceValues, RollingWindowSize);
                    var rollingRewardRate = ComputeRollingAverage(rewardValues, RollingWindowSize);
                    var rollingFinishedRate = ComputeRollingAverage(finishedTrialValues.Select(v => (float?)v).ToArray(), RollingWindowSize);
                    // Categorize trials
                    List<float> rightChoiceX;
                    List<float> leftChoiceX;
                    List<float> rightRewardedX;
                    List<float> rightUnrewardedX;
                    List<float> leftRewardedX;
                    List<float> leftUnrewardedX;
                    List<float> noChoiceX;
                    List<float> rightRewardedAutoX;
                    List<float> rightUnrewardedAutoX;
                    List<float> leftRewardedAutoX;
                    List<float> leftUnrewardedAutoX;

                    CategorizeTrials(TrialOutcomes,
                        out rightChoiceX,
                        out leftChoiceX,
                        out rightRewardedX,
                        out rightUnrewardedX,
                        out leftRewardedX,
                        out leftUnrewardedX,
                        out noChoiceX,
                        out rightRewardedAutoX,
                        out rightUnrewardedAutoX,
                        out leftRewardedAutoX,
                        out leftUnrewardedAutoX
                    );

                    // Extract probability traces
                    var pRewardLeft = ExtractPRewardLeft(TrialOutcomes);
                    var pRewardRight = ExtractPRewardRight(TrialOutcomes);

                    if (Visible)
                    {
                        if (ImPlot.BeginPlot("Trial Outcomes", new Vector2(-1, TopPlotHeight), ImPlotFlags.NoTitle))
                        {
                            double[] tickPositions = new double[] {0, 1};
                            ImPlot.SetupAxisTicks(ImAxis.Y1, ref tickPositions[0], 2, new string[] {"R", "L"});
                            ImPlot.SetupAxisLimits(ImAxis.Y1, -0.5f, 1.5f);
                            ImPlot.SetupLegend(ImPlotLocation.East, ImPlotLegendFlags.Outside);

                            // plot left choices
                            ScatterHorizontalSeries("Left Choice", leftChoiceX.ToArray(), 1.2f, ImPlotMarker.Circle, MarkerSize, MarkerWeight, new Vector4(0, 1, 0, 0), new Vector4(0, 0.7f, 0, 1));

                            // plot left rewarded choices
                            ScatterHorizontalSeries("Left Rewarded", leftRewardedX.ToArray(), 1f, ImPlotMarker.Circle, MarkerSize, MarkerWeight, new Vector4(0, 1, 0, 1), new Vector4(0, 0.7f, 0, 1));

                            // plot right choices
                            ScatterHorizontalSeries("Right Choice", rightChoiceX.ToArray(), -0.2f, ImPlotMarker.Circle, MarkerSize, MarkerWeight, new Vector4(0, 1, 0, 0), new Vector4(0, 0.7f, 0, 1));

                            // plot right rewrded choices
                            ScatterHorizontalSeries("Right Rewarded", rightRewardedX.ToArray(), 0f, ImPlotMarker.Circle, MarkerSize, MarkerWeight, new Vector4(0, 1, 0, 1), new Vector4(0, 0.7f, 0, 1));

                            // plot no choice
                            ScatterHorizontalSeries("No Choice", noChoiceX.ToArray(), 0.5f, ImPlotMarker.Square, MarkerSize, MarkerWeight, new Vector4(0, 0, 0, 0.2f), new Vector4(0, 0, 0, 0.2f));

                            ImPlot.EndPlot();
                        }
                        if (ImPlot.BeginPlot("Trial History", new Vector2(-1, BottomPlotHeight), ImPlotFlags.NoTitle))
                        {
                            double[] tickPositions = new double[] {0, 1};
                            ImPlot.SetupAxisTicks(ImAxis.Y1, ref tickPositions[0], 2);
                            ImPlot.SetupAxisLimits(ImAxis.Y1, -0.5f, 1.5f);
                            ImPlot.SetupLegend(ImPlotLocation.East, ImPlotLegendFlags.Outside);

                            PlotRollingLine("pL", trialIndices, pRewardLeft, new Vector4(1, 0, 0, 1), RollingLineThickness);
                            PlotRollingLine("pR", trialIndices, pRewardRight, new Vector4(0, 0, 1, 1), RollingLineThickness);

                            PlotRollingLine("Choice Fraction", trialIndices, rollingChoiceAvg, new Vector4(0, 0, 0, 1), RollingLineThickness);
                            PlotRollingLine("Reward Rate", trialIndices, rollingRewardRate, new Vector4(0, 1, 0, 1), RollingLineThickness);
                            PlotRollingLine("Finished Rate", trialIndices, rollingFinishedRate, new Vector4(0, 0.5f, 0.5f, 1), RollingLineThickness);

                            ImPlot.EndPlot();
                        }
                        observer.OnNext(value);
                    }
                },
                observer.OnError,
                observer.OnCompleted);
            return source.SubscribeSafe(sourceObserver);
        });
    }

    private void ScatterHorizontalSeries(string label, float[] values, float ySet, ImPlotMarker markerType, float size, float weight, Vector4 fillColor, Vector4 outlineColor)
    {
        float[] ys = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            ys[i] = ySet;
        }

        ImPlot.SetNextMarkerStyle(markerType, size, fillColor, weight, outlineColor);
        ImPlot.PlotScatter(label, ref values[0], ref ys[0], values.Length);
    }

    /// <summary>
    /// Plots a line for rolling averages.
    /// </summary>
    private static void PlotRollingLine(
        string label,
        float[] xValues,
        float[] yValues,
        Vector4 color,
        float lineThickness)
    {
        if (xValues.Length == 0) return;

        ImPlot.SetNextLineStyle(color, lineThickness);
        ImPlot.PlotLine(label, ref xValues[0], ref yValues[0], xValues.Length);
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
    /// Categorizes trials and collects their x positions.
    /// </summary>
    private static void CategorizeTrials(
        List<TrialOutcome> trialList,
        out List<float> rightChoiceX,
        out List<float> leftChoiceX,
        out List<float> rightRewardedX,
        out List<float> rightUnrewardedX,
        out List<float> leftRewardedX,
        out List<float> leftUnrewardedX,
        out List<float> noChoiceX,
        out List<float> rightRewardedAutoX,
        out List<float> rightUnrewardedAutoX,
        out List<float> leftRewardedAutoX,
        out List<float> leftUnrewardedAutoX)
    {
        rightChoiceX = new List<float>();
        leftChoiceX = new List<float>();
        rightRewardedX = new List<float>();
        rightUnrewardedX = new List<float>();
        leftRewardedX = new List<float>();
        leftUnrewardedX = new List<float>();
        noChoiceX = new List<float>();
        rightRewardedAutoX = new List<float>();
        rightUnrewardedAutoX = new List<float>();
        leftRewardedAutoX = new List<float>();
        leftUnrewardedAutoX = new List<float>();

        for (int i = 0; i < trialList.Count; i++)
        {
            var trial = trialList[i];
            float x = i;
            bool isAutoResponse = trial.Trial.IsAutoResponseRight.HasValue;

            if (!trial.IsRightChoice.HasValue)
            {
                noChoiceX.Add(x);
            }
            else if (trial.IsRightChoice.Value)
            {
                if (isAutoResponse)
                {
                    if (trial.IsRewarded)
                    {
                        rightRewardedAutoX.Add(x);
                    }
                    else
                    {
                        rightUnrewardedAutoX.Add(x);
                    }
                }
                else
                {
                    rightChoiceX.Add(x);
                    if (trial.IsRewarded)
                    {
                        rightRewardedX.Add(x);
                    }
                    else
                    {
                        rightUnrewardedX.Add(x);
                    }
                }
            }
            else
            {
                if (isAutoResponse)
                {
                    if (trial.IsRewarded)
                    {
                        leftRewardedAutoX.Add(x);
                    }
                    else
                    {
                        leftUnrewardedAutoX.Add(x);
                    }
                }
                else
                {
                    leftChoiceX.Add(x);
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
    /// Extracts finished trial values (1 if choice was made, 0 if not) from all trials.
    /// </summary>
    private static float[] ExtractFinishedTrialValues(List<TrialOutcome> trialList)
    {
        var values = new float[trialList.Count];
        for (int i = 0; i < trialList.Count; i++)
        {
            values[i] = trialList[i].IsRightChoice.HasValue ? 1.0f : 0.0f;
        }
        return values;
    }
}

