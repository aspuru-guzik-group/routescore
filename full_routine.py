# Partition list of laser molecules by type of synthesis route (iSMC, SNAr, BHA, B-S, S-B)
import partition
# Calculate RouteScores for all iSMC routes
import RS_Base
# Calculate RouteScores for all SNAr routes
import RS_SNAr
# Calculate RouteScores for all BHA routes
import RS_Buch
# Calculate RouteScores for all S-B routes
import RS_SB
# Calculate RouteScores for all B-S routes
import RS_BS
# Add information about manual/automated synthesis to each, and save to RS_xxx.pkl
import man_part
# Save dataframe with only RouteScores
import only_RS
# Update full_props.pkl with new RouteScores
import merge_newRS
