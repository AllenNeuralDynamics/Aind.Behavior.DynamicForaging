public enum Choice
{
    NoChoice,
    RightChoice,
    LeftChoice
}

public enum Category
{
    RightRewardedAuto,
    LeftRewardedAuto,
    RightUnrewardedAuto,
    LeftUnrewardedAuto,
    RightRewarded,
    RightUnrewarded,
    LeftRewarded,
    LeftUnrewarded,
}

public class CategorizedTrial
{
    public Choice ChoiceCategory {get; set;}
    public Category TrialCategory {get; set;}
}