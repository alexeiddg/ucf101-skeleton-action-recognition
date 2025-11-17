UCF101_CLASSES = [
    'ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam',
    'BandMarching', 'BaseballPitch', 'Basketball', 'BasketballDunk', 'BenchPress',
    'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats',
    'Bowling', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BreastStroke', 'BrushingTeeth',
    'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot', 'CuttingInKitchen',
    'Diving', 'Drumming', 'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics',
    'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut', 'Hammering',
    'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
    'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow',
    'JugglingBalls', 'JumpingJack', 'JumpRope', 'Kayaking', 'Knitting',
    'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
    'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf',
    'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar',
    'PlayingTabla', 'PlayingViolin', 'PoleVault', 'PommelHorse', 'PullUps',
    'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing',
    'Rowing', 'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding',
    'Skiing', 'Skijet', 'SkyDiving', 'SoccerJuggling', 'SoccerPenalty',
    'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
    'TaiChi', 'TennisSwing', 'ThrowDiscus', 'TrampolineJumping', 'Typing',
    'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog', 'WallPushups', 'WritingOnBoard',
    'YoYo'
]

IDX_TO_CLASS = {i: classname for i, classname in enumerate(UCF101_CLASSES)}

CLASS_TO_IDX = {classname: i for i, classname in enumerate(UCF101_CLASSES)}


def get_class_name(class_idx):
    return IDX_TO_CLASS.get(class_idx, "Unknown")


def get_class_idx(class_name):
    return CLASS_TO_IDX.get(class_name, -1)