using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using AindDynamicForagingDataSchema;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class DetermineReward
{
    private Random random = new Random();
    public Random MyProperty
    {
        get { return random; }
        set { random = value; }
    }

    public IObservable<bool> Process(IObservable<Tuple<bool?, Trial>> source)
    {
        return source.Select(value =>
        {
            var response = value.Item1;
            var trial = value.Item2;
            if (!response.HasValue)
            {
                return false;
            }
            var p = response.Value ? trial.PRewardRight : trial.PRewardLeft;
            return random.NextDouble() < p;
        });
    }
}
