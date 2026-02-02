using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("Creates a nullable Boolean.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateNullableBoolean
{
    [Description("Value to be propagated")]
    public bool? Value { get; set; }
    public IObservable<bool?> Process()
    {
        return Observable.Return(Value);
    }
    public IObservable<bool?> Process<TSource>(IObservable<TSource> source)
    {
        return source.Select(x => Value);
    }
}
