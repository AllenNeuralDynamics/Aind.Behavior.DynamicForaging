using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;
using AindDynamicForagingDataSchema;
using Bonsai.Expressions;

[Combinator]
[Description("Deserializes a PyObject into a specified .NET type using JSON-backed serialization.")]
[WorkflowElementCategory(ElementCategory.Transform)]

[System.Xml.Serialization.XmlIncludeAttribute(typeof(TypeMapping<Trial>))]
[System.Xml.Serialization.XmlIncludeAttribute(typeof(TypeMapping<TrialOutcome>))]
[System.Xml.Serialization.XmlIncludeAttribute(typeof(TypeMapping<TrialGeneratorSpec>))]
public class DeserializeFromPyObject : SingleArgumentExpressionBuilder
{
    public DeserializeFromPyObject()
    {
        Type = new TypeMapping<Trial>();
    }
    public TypeMapping Type { get; set; }

    public override System.Linq.Expressions.Expression Build(IEnumerable<System.Linq.Expressions.Expression> arguments)
    {
        var typeMapping = (TypeMapping)Type;
        var returnType = typeMapping.GetType().GetGenericArguments()[0];
        return System.Linq.Expressions.Expression.Call(
            typeof(DeserializeFromPyObject),
            "Process",
            new Type[] { returnType },
            Enumerable.Single(arguments));
    }

    private static IObservable<TSerializable> Process<TSerializable>(IObservable<PyObject> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var serialized = value.InvokeMethod("model_dump_json").As<string>();
                return Newtonsoft.Json.JsonConvert.DeserializeObject<TSerializable>(serialized);
            }
        });
    }
}
