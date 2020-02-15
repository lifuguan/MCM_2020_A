# MCM2020 A

#### 感兴趣区域（ROI） 
65°N 10°W | 左上角

50°N 03°E | 右下角

https://ladsweb.modaps.eosdis.nasa.gov/search/order/3/MOD021KM--61,MOD03--61/2014-01-01..2020-02-14/DB/-10,65,3,50

#### 解析方式

Data are stored in ASCII

1. All values are integers
2. Temperatures are stored as degrees C * 100
3. 100% sea-ice-covered gridboxes are flagged as -1000
4. Land squares are set to -32768

The day, month and year are stored at the start of each month. The day simply tells
you on which day the month starts.

Data Array (360x180)
Item (  1,  1) stores the value for the 1-deg-area centred at 179.5W and 89.5N
Item (360,180) stores the value for the 1-deg-area centred at 179.5E and 89.5S

```bash
          ----- ----- -----
         |     |     |     |
         | DAY | MON | YR  |
         |_____|_____|_____|____________________________
     90N |(1,1)                                         |
         |                                              |
         |                                              |
         |                                              |
         |                                              |
         |(1,90)                                        |
     Equ |                                              |
         |(1,91)                                        |
         |                                              |
         |                                              |
         |                                              |
         |                                              |
     90S |(1,180)______________________________(360,180)|
          180W                 0                    180E
```
#### 注意事项

1°经度差对应的东西方向的距离是与其纬度有密切关系的。
赤道上经度相差1°对应的弧长大约是111千米
- 20° 104公里
- 26° 100公里
- 30° 96公里
- 36° 90公里
- 40° 85公里
- 44° 80公里
- 51° 70公里

#### 截取数据
