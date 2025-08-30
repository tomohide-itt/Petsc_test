Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 1};
Line(4) = {1, 2};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

//=== 物理グループ
// 領域（2Dドメイン）
Physical Surface( "domain", 100 ) = {1};

// 境界（1D エッジ）
Physical Curve( "right", 1 ) = {1};
Physical Curve( "top", 2 ) = {2};
Physical Curve( "left", 3 ) = {3};
Physical Curve( "bottom", 4 ) = {4};

// 節点（0D 頂点）
Physical Point( "corner1", 11 ) = {1};
Physical Point( "corner2", 12 ) = {2};
Physical Point( "corner3", 13 ) = {3};
Physical Point( "corner4", 14 ) = {4};
