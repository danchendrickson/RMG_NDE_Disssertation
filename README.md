# RMG_NDE_Disssertation
Code to support non-destructive evaluation of RMG cranes PhD dissertation
## Overal Goals for Crane Data:
1. Show comparison of Beta Wavelet with other standard for catagorizing fualts
1. Compare standard Wavelet, with Fingerprint, with video, with image, with expert system, with straight signal

## To do list for Beta Wavelet:
1. make it work
1. make real time analysis
1. compare against all options with real time analysis

## To Do list for Crane Data Analysis:
1. <font color = 'blue'>Choose Random set from Repository</font>
1. <font color = 'blue'>Parse Name to get accepted output vector</font>
1. Filter out non-move data set
1. Train Models on data
    * Data Prep:
      * <font color = 'blue'>Try un smoothed</font>
      * <font color = 'blue'>Try smoothed with rolling average</font>
      * <font color = 'blue'>Make fingerprints by dimmension</font>
      * <font color = 'blue'>Make fingerprints by r</font>
      * <font color = 'blue'>try smooth with kalman</font>
      * <font color = 'blue'>set up wavelet low-pass filter</font>
    * Model TypesVideo and Image
      * Video:
        * <font color = 'blue'>Rolling, 3x100, 3x changing frame sizes</font>
        * <font color = 'blue'>overlapping segments, 3x?? hoping down line</font>
        * <font color = 'blue'> rolling fingerprints</font>
          * Try many fingerprints
          * look at Beta, and classification as it is going
          * Features from live fingerprints
      * Image:
        * Fingerprint
        * 3 x 60k image
1. Examine differences of results

## Rail LASER
1. Start data
1. Figure frequency needed
1. Specific test cases
    * 'Singing track'
    * Over center anchor rise
    * Old, new, ground, rough, tamped, shakey

## EFIT and related
1. <font color = 'blue'>Write code</font>
1. Get code working
1. GPU parrallelization
1. Inner simulation space bundary conditions
1. Compare LASER measurement to EFIT
