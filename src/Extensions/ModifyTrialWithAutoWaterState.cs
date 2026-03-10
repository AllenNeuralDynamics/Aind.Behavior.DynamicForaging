using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Newtonsoft.Json;

namespace AindDynamicForagingDataSchema
{
    [Combinator]
    [Description("Modifies the trial object based on the auto water state.")]
    [WorkflowElementCategory(ElementCategory.Transform)]
    public class ModifyTrialWithAutoWaterState
    {
        public IObservable<Trial> Process(IObservable<Tuple<Trial, AutoWaterState>> source)
        {
            return source.Select(value =>
            {
                var autoWaterState = value.Item2;
                if (autoWaterState == null || !autoWaterState.HasValue) return value.Item1;

                // the copy-constructor is protected, we use a JSON round-trip to create a deep copy of the trial object
                var newTrial = JsonConvert.DeserializeObject<Trial>(JsonConvert.SerializeObject(value.Item1));
                if (autoWaterState.IsRight)
                {
                    newTrial.PRewardRight = 1.0f;
                    newTrial.IsAutoResponseRight = true;
                    return newTrial;
                }
                if (autoWaterState.IsLeft)
                {
                    newTrial.PRewardLeft = 1.0f;
                    newTrial.IsAutoResponseRight = false;
                    return newTrial;
                }
                throw new InvalidOperationException("AutoWaterState must be either right or left.");
            }
            );
        }
    }
}
