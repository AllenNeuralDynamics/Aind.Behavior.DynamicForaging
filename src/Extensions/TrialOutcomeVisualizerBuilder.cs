﻿
using Bonsai.Expressions;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Linq.Expressions;
using Bonsai;
using AindDynamicForagingDataSchema;


[TypeVisualizer(typeof(TrialOutcomeVisualizer))]
[WorkflowElementCategory(ElementCategory.Combinator)]
[Description("Visualizes a table of recent Trial properties.")]
public class TrialOutcomeVisualizerBuilder : SingleArgumentExpressionBuilder
{
    private float fontSize = 16.0f;
    public float FontSize
    {
        get { return fontSize; }
        set { fontSize = value; }
    }

    /// <inheritdoc/>
    public override Expression Build(IEnumerable<Expression> arguments)
    {
        var source = arguments.First();

        return Expression.Call(typeof(TrialOutcomeVisualizerBuilder), "Process", null, source);
    }

    static IObservable<TrialOutcome> Process(IObservable<TrialOutcome> source)
    {
        return source;
    }
}

