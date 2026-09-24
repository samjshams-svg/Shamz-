# Super Bowl XLIX: Lynch gets the ball  
  
## Part 1: the game  
  
NE at SEA (2014_21_NE_SEA), play 4205  
  real play : (:26) (Shotgun) 3-R.Wilson pass short right intended for 83-R.Lockette INTERCEPTED by 21-M.Butler at NE -1. 21-M.Butler to NE 2 for 3 yards (83-R.Lockette). PENALTY on NE, Unsportsmanlike Conduct, 1 yard, enforced at NE 2.  
  before    : SEA 24 - NE 28 | Q4 0:26 | SEA ball, 2&1 at NE 1 | timeouts SEA:1 NE:2  
  edit      : run  
  after     : SEA 24 - NE 28 | Q4 0:26 | SEA ball, 2&1 at NE 1 (next play: run) | timeouts SEA:1 NE:2  
  
Win probability after the edited play (10,000 simulations):  
  SEA   79.1%  (95% CI 78.3%-79.9%)  
  NE    20.9%  (95% CI 20.1%-21.7%)  
  
For comparison (SEA win probability):  
  nflfastR before the snap  63.3% | after the real play   6.0%  
  this model before snap    78.0% | after the real play   0.0%  
  real final: NE 28, SEA 24  
  
## Part 2: the next seasons  
  
10,000 simulated histories per world. Elo: K=20, HFA=48, regression=0.50, playoff x1.  
Ratings just before the changed game: SEA 1763, NE 1716  
  
season team                   Wins    Playoffs    Division Conf. title  Super Bowl  
----------------------------------------------------------------------------------  
2014   SEA   actual           12.0         yes         yes         yes           -  
             no change        12.0      100.0%      100.0%      100.0%        0.0%  
             with edit        12.0      100.0%      100.0%      100.0%       79.1%  
2014   NE    actual           12.0         yes         yes         yes         yes  
             no change        12.0      100.0%      100.0%      100.0%      100.0%  
             with edit        12.0      100.0%      100.0%      100.0%       20.9%  
2015   SEA   actual           10.0         yes           -           -           -  
             no change        10.3       68.2%       50.5%       19.5%       11.1%  
             with edit        10.5       71.0%       53.4%       21.6%       12.2%  
2015   NE    actual           12.0         yes         yes           -           -  
             no change        10.5       70.5%       51.6%       18.6%       10.5%  
             with edit        10.3       67.2%       48.5%       16.7%        9.1%  
2016   SEA   actual           10.5         yes         yes           -           -  
             no change         9.2       53.2%       37.0%       11.6%        6.4%  
             with edit         9.4       55.1%       38.5%       12.3%        6.8%  
2016   NE    actual           14.0         yes         yes         yes         yes  
             no change         9.1       52.7%       38.0%       10.9%        6.2%  
             with edit         9.0       50.8%       36.6%       10.3%        5.6%  
  
Other teams whose playoff odds moved most:  
  2015 MIA   29.0% ->  29.9% (+0.9%)  
  2015 ARI   37.3% ->  36.5% (-0.8%)  
  2015 BUF   43.7% ->  44.5% (+0.8%)  
  2015 SF    32.6% ->  31.8% (-0.7%)  
  2015 NYJ   23.0% ->  23.6% (+0.6%)  
  2016 PIT   40.9% ->  41.5% (+0.6%)  
  2016 GB    49.2% ->  48.7% (-0.5%)  
  2016 SF    36.3% ->  35.9% (-0.4%)  
  2016 DEN   47.4% ->  47.8% (+0.4%)  
  2016 IND   51.8% ->  51.4% (-0.4%)
