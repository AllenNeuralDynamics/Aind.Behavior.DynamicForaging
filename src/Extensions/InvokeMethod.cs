using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;
using Python.Runtime;
using Newtonsoft.Json;

[Combinator]
[Description("Invokes a method on a Python object with string arguments converted from the input values.")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class InvokeMethod
{
    private string methodName = "update";
    [Description("The name of the method to invoke on the Python object.")]
    public string MethodName
    {
        get { return methodName; }
        set { methodName = value; }
    }

    private bool attemptSerialization = true;
    [Description("Specifies whether to attempt JSON serialization for argument types that are not directly convertible to PyObject.")]
    public bool AttemptSerialization
    {
        get { return attemptSerialization; }
        set { attemptSerialization = value; }
    }

    public IObservable<PyObject> Process(IObservable<PyObject> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.GetAttr(methodName);
                var args = new PyObject[] { };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1>(IObservable<Tuple<PyObject, T1>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2>(IObservable<Tuple<PyObject, Tuple<T1, T2>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2, T3>(IObservable<Tuple<PyObject, Tuple<T1, T2, T3>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item3, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2, T3, T4>(IObservable<Tuple<PyObject, Tuple<T1, T2, T3, T4>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item3, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item4, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2, T3, T4, T5>(IObservable<Tuple<PyObject, Tuple<T1, T2, T3, T4, T5>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item3, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item4, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item5, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2, T3, T4, T5, T6>(IObservable<Tuple<PyObject, Tuple<T1, T2, T3, T4, T5, T6>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item3, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item4, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item5, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item6, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    public IObservable<PyObject> Process<T1, T2, T3, T4, T5, T6, T7>(IObservable<Tuple<PyObject, Tuple<T1, T2, T3, T4, T5, T6, T7>>> source)
    {
        return source.Select(value =>
        {
            using (Py.GIL())
            {
                var method = value.Item1.GetAttr(methodName);
                var args = new PyObject[] {
                    ConvertToPyObject(value.Item2.Item1, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item2, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item3, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item4, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item5, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item6, attemptSerialization),
                    ConvertToPyObject(value.Item2.Item7, attemptSerialization)
                };
                return method.Invoke(args);
            }
        });
    }

    private static PyObject ConvertToPyObject(object value, bool attemptSerialization)
    {
        using (Py.GIL())
        {
            if (value == null)
                return PythonEngine.Eval("None");
            if (value is PyObject)
                return (PyObject)value;
            if (value is string)
                return new PyString((string)value);
            if (value is int)
                return new PyInt((int)value);
            if (value is long)
                return new PyInt((long)value);
            if (value is short)
                return new PyInt((short)value);
            if (value is byte)
                return new PyInt((byte)value);
            if (value is uint)
                return new PyInt((uint)value);
            if (value is ulong)
                return new PyInt(checked((long)(ulong)value));
            if (value is float)
                return new PyFloat((float)value);
            if (value is double)
                return new PyFloat((double)value);
            if (value is decimal)
                return new PyFloat((double)(decimal)value);
            if (value is bool)
                return PythonEngine.Eval((bool)value ? "True" : "False");
            if (attemptSerialization)
            {
                var json = JsonConvert.SerializeObject(value);
                return new PyString(json);
            }
            throw new NotSupportedException("Cannot convert value of type " + value.GetType() + " to a Python object.");
        }
    }
}
