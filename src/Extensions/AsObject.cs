using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Newtonsoft.Json.Linq;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class AsObject
{
    public IObservable<object> Process(IObservable<IDictionary<string, string>> source)
    {
        return source.Select(value => JObject.FromObject(value));
    }
}
