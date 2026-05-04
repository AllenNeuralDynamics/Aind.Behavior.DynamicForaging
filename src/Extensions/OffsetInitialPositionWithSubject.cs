using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using AindDynamicForagingDataSchema;
using AllenNeuralDynamics.AindManipulator;
using Newtonsoft.Json;

[Combinator]
[Description("Offsets the initial position of the manipulator with the subject's calibration.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class OffsetInitialPositionWithSubject
{
    public IObservable<AindManipulatorCalibration> Process(IObservable<DynamicForagingAindManipulator> source)
    {
        return source.Select(value =>
        {
            // Make a quick deep-copy just in case.
            var calibration = JsonConvert.DeserializeObject<AindManipulatorCalibration>(JsonConvert.SerializeObject(value.Calibration));
            if (calibration == null)
            {
                throw new InvalidOperationException("Manipulator does not contain a valid initial position.");
            }
            calibration.InitialPosition = calibration.InitialPosition + value.SubjectOffset;
            return calibration;
        }
        );
    }
}
