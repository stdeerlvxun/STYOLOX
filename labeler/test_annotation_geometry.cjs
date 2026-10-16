const assert=require('node:assert/strict');
const {imagePoint,screenPoint,imageRect}=require('./web/annotations.js');
for(const [width,height,imageWidth,imageHeight]of [[800,600,1600,900],[400,600,600,1200],[900,400,1000,1000]]){
  for(const rotation of [0,90,180,270])for(const zoom of [.5,1,3,8]){
    const view={width,height,imageWidth,imageHeight,rotation,zoom,panX:73,panY:-31};
    for(const [x,y]of [[0,0],[.1,.2],[.5,.5],[1,1],[.999,.001]]){
      const screen=screenPoint(x,y,view),point=imagePoint(screen.x,screen.y,view);
      assert.ok(Math.abs(point.x-x)<1e-10);assert.ok(Math.abs(point.y-y)<1e-10);
    }
    assert.ok(imageRect(view).width<=width&&imageRect(view).height<=height);
  }
}
assert.equal(imagePoint(0,0,{width:800,height:600,imageWidth:1600,imageHeight:900,rotation:0,zoom:1,panX:0,panY:0}).inside,false);
console.log('Localization coordinates: letterboxing, 4 rotations, zoom, pan, and boundaries passed.');
