================================================
Script to get Historic NWCC SNOTEL/SCAN Data
================================================
#!/bin/sh
#Script: gethistoric.sh
#Purpose: get historic data for a station
#Input: station_id, report_type, series year mon <optional day>
#Output: csv-formatted file
#Maggie Dunklee, 2010, National Water Climate Center, All Data is Provisional
#
#if  [[ $# -lt 5  ]] 
#then
#    echo "Script to get historic data"
#    echo "usage: $0 STATION   REPORT_TYPE   TIME_SERIES   YEAR   MONTH   DAY"
#    echo "where: "
#    echo "      STATION is numeric is 2095, 302, etc"
#    echo "      REPORT_TYPE is  SOIL (soil temp and moisture) SCAN (standard scan), ALL (all), WEATHER (atmospheric)"
#    echo "      TIME_SERIES is Daily or Hourly or Hour:HH"
#    echo "      YEAR is YYYY"
#    echo "      MONTH is MM or CY (calendar year), or WY  (water year)"
#    echo "      Optional DAY is DD"
#    exit
#fi


statids=(15 581 674 808 2001 2002 2003 2004 2005 2006 2008 2009 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2024 2025 2026 2027 2028 2030 2031 2032 2033 2034 2035 2036 2037 2038 2039 2041 2042 2043 2045 2046 2047 2048 2049 2050 2051 2052 2053 2055 2056 2057 2060 2061 2062 2063 2064 2066 2067 2068 2069 2070 2072 2073 2074 2075 2076 2077 2078 2079 2082 2083 2084 2085 2086 2087 2088 2089 2090 2091 2092 2093 2094 2096 2097 2099 2101 2102 2104 2105 2106 2107 2108 2109 2110 2111 2112 2113 2114 2115 2116 2117 2118 2119 2120 2121 2122 2123 2125 2126 2127 2128 2129 2130 2131 2132 2133 2134 2135 2136 2137 2138 2139 2140 2141 2142 2144 2145 2146 2147 2148 2149 2150 2151 2152 2153 2154 2155 2156 2157 2158 2159 2160 2161 2162 2163 2164 2165 2166 2167 2168 2169 2171 2172 2173 2174 2175 2176 2177 2178 2179 2180 2181 2182 2183 2184 2185 2186 2187 2188 2189 2190 2191 2192 2193 2194 2195 2196 2197 2198 2199 2200 2201 2202 2203 2204 2205 2206 2207 2212 2213 2214 2215 2216 2217 2218 2219 2220 2223 2224 2225 2226 2227 2228 2229 2230 2231)


#export STATION=$1
#export REPORT=$2
#export SERIES=$3
#export YEAR=$4
#export MONTH=$5
#export DAY=$6

export REPORT=SCAN
export SERIES=Daily
export YEAR=2021
export MONTH=CY
export INTERVALTYPE=Historic
export DAY=

echo $REPORT
echo $SERIES
echo $YEAR
echo $MONTH
echo $DAY

for stid in ${statids[@]}; do
   echo $stid
   echo "https://wcc.sc.egov.usda.gov/nwcc/view?intervalType=$INTERVALTYPE+&report=$REPORT&timeseries=$SERIES&format=copy&sitenum=$stid&year=$YEAR&month=$MONTH&day=$DAY"
   curl "https://wcc.sc.egov.usda.gov/nwcc/view?intervalType=$INTERVALTYPE+&report=$REPORT&timeseries=$SERIES&format=copy&sitenum=$stid&year=$YEAR&month=$MONTH&day=$DAY" -o $stid-$YEAR$MONTH$DAY.csv
done
exit 0

