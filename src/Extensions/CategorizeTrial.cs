using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Bonsai.Harp;
using AindDynamicForagingDataSchema;
using System.Xml.Serialization;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CategorizeTrial
{
    public IObservable<Timestamped<CategorizedTrial>> Process(IObservable<Timestamped<TrialOutcome>> source)
    {
        return source.Select(value =>
        {
            var trial = value.Value;
            bool isAutoResponse = trial.Trial.IsAutoRewardRight.HasValue;
            var categorizedTrial = new CategorizedTrial();
            if (!trial.IsRightChoice.HasValue)
            {
                categorizedTrial.ChoiceCategory = Choice.NoChoice;
            }
            else if (trial.IsRightChoice.Value)
            {
                if (isAutoResponse)
                {
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = Category.RightRewardedAuto;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = Category.RightUnrewardedAuto;
                    }
                }
                else
                {
                    categorizedTrial.ChoiceCategory = Choice.RightChoice;
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = Category.RightRewarded;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = Category.RightUnrewarded;
                    }
                }
            }
            else
            {
                if (isAutoResponse)
                {
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = Category.LeftRewardedAuto;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = Category.LeftUnrewardedAuto;
                    }
                }
                else
                {
                    categorizedTrial.ChoiceCategory = Choice.LeftChoice;
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = Category.LeftRewarded;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = Category.LeftUnrewarded;
                    }
                }
            }


            return new Timestamped<CategorizedTrial>(categorizedTrial, value.Seconds);
        });
    }
}



