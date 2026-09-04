using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Harp;
using AindDynamicForagingDataSchema;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class ExtractChoiceValues
{
    public IObservable<float?> Process(IObservable<Timestamped<TrialOutcome>> source)
    {
        return source.Select(value => {
            var choice = value.Value.IsRightChoice;
            if (choice.HasValue)
            {
                return choice.Value ? 1.0f : 0.0f;
            } else
            {
                return (float?)null;
            }
        });
    }
}
