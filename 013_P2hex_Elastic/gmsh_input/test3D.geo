// gmsh test3D.geo -3 -order 2 -format msh2 -clmax 1.0 -clmin 1.0 -o test3D_c.msh
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {1, 0, 1};
Point(7) = {1, 1, 1};
Point(8) = {0, 1, 1};
//
Line(1) = {2, 3};
Line(2) = {3, 4};
Line(3) = {4, 1};
Line(4) = {1, 2};
Line(5) = {2, 6};
Line(6) = {6, 7};
Line(7) = {7, 3};
Line(8) = {7, 8};
Line(9) = {8, 4};
Line(10) = {8, 5};
Line(11) = {5, 6};
Line(12) = {5, 1};
//
Curve Loop(1) = {5, 6, 7, -1};
Curve Loop(2) = {7, 2, -9, -8};
Curve Loop(3) = {3, -12, -10, 9};
Curve Loop(4) = {5, -11, 12, 4};
Curve Loop(5) = {1, 2, 3, 4};
Curve Loop(6) = {6, 8, 10, 11};
//
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
//
Surface Loop(1) = {1, 4, 6, 2, 5, 3};
Volume(1) = {1};

//=== 物理グループ
// 領域（3Dドメイン）
Physical Volume( "domain", 100 ) = {1};

// 境界（2D 面）
Physical Surface( "right", 1 ) = {1};
Physical Surface( "back", 5 ) = {2};
Physical Surface( "left", 3 ) = {3};
Physical Surface( "front", 6 ) = {4};
Physical Surface( "bottom", 4 ) = {5};
Physical Surface( "top", 2 ) = {6};
//
Transfinite Line "*" = 11 Using Bump 1.0;
Transfinite Surface "*";
Recombine Surface "*";
Transfinite Volume "*";
