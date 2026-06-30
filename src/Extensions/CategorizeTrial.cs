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
            bool isAutoResponse = trial.Trial.IsAutoResponseRight.HasValue;
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
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.RightRewardedAuto;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.RightUnrewardedAuto;
                    }
                }
                else
                {
                    categorizedTrial.ChoiceCategory = Choice.RightChoice;
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.RightRewarded;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.RightUnrewarded;
                    }
                }
            }
            else
            {
                if (isAutoResponse)
                {
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.LeftRewardedAuto;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.LeftUnrewardedAuto;
                    }
                }
                else
                {
                    categorizedTrial.ChoiceCategory = Choice.LeftChoice;
                    if (trial.IsRewarded)
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.LeftRewarded;
                    }
                    else
                    {
                        categorizedTrial.TrialCategory = CategorizedTrial.Category.LeftUnrewarded;
                    }
                }
            }


            return new Timestamped<CategorizedTrial>(categorizedTrial, value.Seconds);
        });
    }
}



