| Truy vấn | Lớp | Histogram màu – top 5 | Hit (số đúng) | CLIP embedding – top 5 | Hit (số đúng) |
|---|---|---|---|---|---|
| q01.jpg | bicycle | motorbike_07, bus_10, motorbike_09, car_07, bus_08 | ✗ (0/5) | **bicycle_03**, **bicycle_09**, **bicycle_05**, bus_01, **bicycle_02** | ✓ (4/5) |
| q02.jpg | bicycle | car_05, motorbike_05, **bicycle_03**, bus_08, boat_01 | ✓ (1/5) | **bicycle_04**, **bicycle_09**, **bicycle_10**, **bicycle_06**, **bicycle_08** | ✓ (5/5) |
| q03.jpeg | boat | bicycle_10, car_01, bus_04, bus_02, bicycle_09 | ✗ (0/5) | **boat_03**, **boat_09**, **boat_02**, **boat_10**, **boat_01** | ✓ (5/5) |
| q04.jpg | boat | **boat_01**, bus_08, bus_10, car_03, bicycle_01 | ✓ (1/5) | **boat_10**, **boat_04**, bicycle_09, **boat_03**, car_05 | ✓ (3/5) |
| q05.jpg | bus | car_03, **bus_09**, **bus_06**, bicycle_04, bicycle_03 | ✓ (2/5) | **bus_07**, **bus_10**, **bus_09**, **bus_02**, **bus_03** | ✓ (5/5) |
| q06.jpg | bus | **bus_09**, **bus_03**, **bus_01**, car_03, **bus_02** | ✓ (4/5) | **bus_09**, **bus_03**, **bus_07**, **bus_10**, **bus_08** | ✓ (5/5) |
| q07.jpg | car | motorbike_04, motorbike_05, motorbike_02, **car_10**, boat_01 | ✓ (1/5) | **car_10**, **car_08**, **car_01**, motorbike_04, bicycle_05 | ✓ (3/5) |
| q08.jpg | car | bus_01, bus_07, **car_06**, bus_09, **car_10** | ✓ (2/5) | **car_01**, motorbike_06, motorbike_02, **car_10**, **car_08** | ✓ (3/5) |
| q09.jpg | motorbike | bicycle_09, **motorbike_06**, bicycle_10, bicycle_06, car_08 | ✓ (1/5) | **motorbike_03**, **motorbike_04**, **motorbike_06**, **motorbike_02**, **motorbike_07** | ✓ (5/5) |
| q10.jpg | motorbike | **motorbike_02**, **motorbike_05**, **motorbike_03**, bicycle_08, **motorbike_09** | ✓ (4/5) | **motorbike_08**, **motorbike_01**, bicycle_05, car_10, **motorbike_10** | ✓ (3/5) |

**Hit@5** — Histogram màu: **8/10** | CLIP embedding: **10/10**

_ID in đậm = kết quả đúng (cùng lớp với truy vấn)._