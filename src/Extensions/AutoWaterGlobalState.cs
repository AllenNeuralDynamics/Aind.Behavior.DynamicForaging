using Bonsai;
using System;
using System.ComponentModel;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("Creates an AutoWaterGlobalState instance.")]
[WorkflowElementCategory(ElementCategory.Source)]
public class CreateAutoWaterState
{
    private bool? value = null;
    [Description("Initial value of the AutoWaterGlobalState instance.")]

    public bool? Value
    {
        get { return value; }
        set { this.value = value; }
    }

    public IObservable<AutoWaterState> Process()
    {
        return Observable.Return(new AutoWaterState(Value));
    }

    public IObservable<AutoWaterState> Process<TSource>(IObservable<TSource> source)
    {
        return source.Select(x => new AutoWaterState(Value));
    }
}

public class AutoWaterState
{
    public AutoWaterState(bool? initialValue)
    {
        IsAutoWaterRight = initialValue;
    }
    private object _lock = new object();
    public bool? IsAutoWaterRight {get; private set; }
    public bool IsLeft { get { return IsAutoWaterRight.HasValue ? !IsAutoWaterRight.Value : false; } }
    public bool IsRight { get { return IsAutoWaterRight.HasValue ? IsAutoWaterRight.Value : false; } }

    public bool HasValue { get { return IsAutoWaterRight.HasValue; } }

    public AutoWaterState Reset()
    {
        lock(_lock)
        {
            IsAutoWaterRight = null;
        }
        return this;
    }

    public AutoWaterState SetRight()
    {
        lock(_lock)
        {
            IsAutoWaterRight = true;
        }
        return this;
    }

    public AutoWaterState SetLeft()
    {
        lock(_lock)
        {
            IsAutoWaterRight = false;
        }
        return this;
    }

    public AutoWaterState ToggleRight()
    {
        lock(_lock)
        {
            if (!IsAutoWaterRight.HasValue)
            {
                IsAutoWaterRight = true;
            }
            else
            {
                IsAutoWaterRight = IsAutoWaterRight.Value ? null : (bool?)true;
            }
        }
        return this;
    }

    public AutoWaterState ToggleLeft()
    {
        lock(_lock)
        {
            if (!IsAutoWaterRight.HasValue)
            {
                IsAutoWaterRight = false;
            }
            else
            {
                IsAutoWaterRight = IsAutoWaterRight.Value ? (bool?)false : null;
            }
        }
        return this;
    }
}
