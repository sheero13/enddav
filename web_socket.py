# 1.Line-Stacked column: X-Traffic, Y-Traffic, line-Traffic
# 2.Stacked-area : X-Traffic,Y-Session,Page view,Time on page 
# 3. Pie - legend-traffic,values-conversion , legend nd values - traffic
# 4. Donut - legend nd values - traffic
# 5. Line - X-traffic,Y-Bounce rate
# 6. Tree map : Category-Traffic,Values - conversion
# 7.Ribbon : X-Traffic,Y-Time on page 
# 8. Clustered bar chart : Y - Traffic, X-Page view,prev,BOunce

# Line-Stacked Column (X-Traffic, Y-Traffic, line-Traffic) → Shows the overall trend of website traffic over time;
#  helps identify peak traffic days and sudden drops.

# Stacked Area (X-Traffic, Y-Session, Page view, Time on page) → Visualizes the contribution of sessions, 
# page views, and time spent on site to total engagement; reveals which metric drives engagement most.

# Pie (Legend-Traffic, Values-Conversion) → Shows the proportion of conversions coming from different traffic
#  sources or segments; highlights the most effective source.

# Donut (Legend & Values-Traffic) → Similar to pie chart; visualizes traffic distribution across 
# segments; useful for quickly spotting dominant traffic sources.

# Line (X-Traffic, Y-Bounce Rate) → Shows changes in bounce rate over time; can detect anomalies where
#  users leave more often, indicating possible UX issues or irrelevant content.

# Tree Map (Category-Traffic, Values-Conversion) → Highlights which traffic categories contribute most 
# to conversions; identifies high-performing segments.

# Ribbon (X-Traffic, Y-Time on Page) → Visualizes ranking changes in time spent on site across 
# traffic sources over time; helps identify improving or declining engagement.

# Clustered Bar Chart (Y-Traffic, X-Page view, Prev, Bounce) → Compares traffic, page views, previous visits,
#  and bounce rate across days or segments; identifies patterns like high traffic but high bounce rate, indicating low engagement quality.
#---------------------------------------------------------------------------------------------------
# DAX 

# TotalLeads = [NumDealsPurchases] + [NumWebPurchases] + [NumCatalogPurchases] + [NumStorePurchases]

# Campaign = 
# IF([AcceptedCmp1] = 1, "Campaign 1",
# IF([AcceptedCmp2] = 1, "Campaign 2",
# IF([AcceptedCmp3] = 1, "Campaign 3",
# IF([AcceptedCmp4] = 1, "Campaign 4",
# IF([AcceptedCmp5] = 1, "Campaign 5","None")))))


# 1. Clustered column: X-Campaign, Y-TotalLeads
# 2. Clustered column: X-TotalLeads bins, Y-Count, Legend-Campaign
# 3. Clustered column: X-Campaign, Y-TotalLeads (Sum)
# 4. Clustered column: X-Campaign, Y-LeadsPerContact
# 5. Scatter: X-Z\_CostContact, Y-TotalLeads, Legend-Campaign
# 6. Clustered column: X-Campaign, Y-ConversionRate
# 7. Stacked column: X-Campaign, Y-NumWebPurchases,NumCatalogPurchases,NumStorePurchases,NumDealsPurchases
# 8. Line: X-Dt\_Customer, Y-TotalLeads, Legend-Campaign

# 1. Compare median leads and distribution across campaigns; identify outliers.
# 2. Frequency distribution of leads per campaign; shows which campaign generates more leads overall.
# 3. Identify campaigns with highest investment.
# 4. Cost per Lead → Campaign with lowest cost per lead = most efficient.
# 5. Scatter Plot Leads vs Budget → Campaigns providing better ROI (more leads per budget).
# 6. Conversion Rate → Campaign with highest conversion efficiency.
# 7. Stacked Bar Leads by Channel → Identify which channels contribute most to leads in each campaign.
# 8. Line Chart Leads Trend → Identify trends, peaks, or seasonal effects in campaign efficiency.
