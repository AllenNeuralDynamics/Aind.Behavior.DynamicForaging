using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using AindDynamicForagingDataSchema;
using Python.Runtime;
using Newtonsoft.Json;

[Combinator]
[Description("Resolves a trial generator from the given specifications and Python module. Returns the instantiated trial generator.")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class ResolveGenerator
{
    public IObservable<PyObject> Process(IObservable<Tuple<TrialGeneratorSpec, PyModule>> source)
    {
        return source.Select(value =>
        {
            var specs = value.Item1;
            var module = value.Item2;
            using (Py.GIL())
            {
                var method = module.GetAttr("resolve_generator");
                string serializedSpec = JsonConvert.SerializeObject(specs);
                var pySpec = new PyString(serializedSpec);
                var trialGenerator = method.Invoke(new PyObject[] { pySpec });
                return trialGenerator;
            }
        }).Take(1);
    }
}
