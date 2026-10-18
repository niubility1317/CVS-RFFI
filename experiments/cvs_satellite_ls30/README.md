# As CE-only label smoothing, fixed E30

Single B01 seed2026092701:strict originalP43 ownAsFINAL20 student+ownEMA plus10 epochs/2220 updates, cumulative30/6660. Original NoAs30/As30 frozen readouts are reused.

The only change is target smoothing .05 rather than .01 on the existing selectedAs64 within mixed128. Its complementary ordinary64,common128,raw128,U and native smoothing remain .01. Original mixed symmetricKL,ORD128 SupCon,6L/768NN and loss scale1.25 remain. This is target entropy,not the closed CE-temperature or CE-weight recipes.

No extra model forward is added. Correction telemetry records4 extra CE operations on256 logit packets perstep,CPU wall and scalarization time separately without explicit CUDA synchronization. Each RDrop pair reuses one IQ draw. Ordinary augmentation remains original20args frozen13;channel/SupCon use cumulative21..30.

Streams:reference_steps,shape_steps,unit_steps,satls_steps and training_sequence,all gzipJSONL. Native metrics are exactly7 concat and7 satls fields from schema.py;concat total is full6L. Extra_CE/RD in reference_steps are the new actual mixed values;old/new values and CE-only delta are in satls_steps.

Launcher first step performs one real own20 checkpoint/noquery2NN plus one actual6L768NN synthetic probe,770NN total,then immediately trains. No local Torch/model claim is made when unavailable. Scientific source failure still completes all7 fixed predictions and independent truthlast scoring. Source criteria are fullV clean newAs-originalNoAs>=-.1pp and two-drawmidlow newAs-originalAs>=+.5pp;no automatic E50 or epsilon scan.
