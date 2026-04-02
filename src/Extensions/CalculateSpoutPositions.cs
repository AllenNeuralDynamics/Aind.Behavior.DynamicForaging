using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using AllenNeuralDynamics.AindManipulator;

[Combinator]
[Description("Calculates the retracted and extended positions of the spouts based on the manipulator position and a specified distance.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class CalculateSpoutPositions
{
    public double SpoutDistance { get; set; }

    public IObservable<SpoutPositions> Process(IObservable<Tuple<ManipulatorPosition, ManipulatorPosition>> source)
    {
        return source.Select(value =>
        {
            var initialPosition = value.Item1;
            var currentPosition = value.Item2;
            return new SpoutPositions
            {
                Retracted = new ManipulatorPosition()
                {
                    X = currentPosition.X,
                    Y1 = initialPosition.Y1 - SpoutDistance,
                    Y2 = initialPosition.Y2 - SpoutDistance,
                    Z = currentPosition.Z
                },
                Extended = new ManipulatorPosition()
                {
                    X = currentPosition.X,
                    Y1 = initialPosition.Y1,
                    Y2 = initialPosition.Y2,
                    Z = currentPosition.Z
                },
            };
        });
    }
}

public class SpoutPositions
{
    public ManipulatorPosition Retracted { get; set; }
    public ManipulatorPosition Extended { get; set; }
}
