/* OnCue: one deterministic timeline shared by the interactive player and video exporter. */
(function(root,factory){if(typeof module==='object'&&module.exports)module.exports=factory();else root.createOnCueFilm=factory()})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  return function createFilm(ctx,images,S,A){
    const W=1000,H=562.5;
    const C={ink:'#1c2e3e',orange:'#e5562e',blue:'#3f82c1',green:'#2e9671',red:'#cf5256',gray:'#7f8d98',paper:'#faf8f5'};
    const chapters=[
      {name:'Scene',title:'Recorded scene',duration:3.5},
      {name:'Contact',title:'Fix contact C',duration:4.5},
      {name:'Exposure',title:'Find exposure E',duration:8},
      {name:'Starts',title:'Build starts and history',duration:7},
      {name:'History',title:'Check hidden approach',duration:10},
      {name:'Rank',title:'Rank plausible starts',duration:4},
      {name:'Generate',title:'Generate target motion',duration:6},
      {name:'Replay',title:'Replay the test',duration:6},
      {name:'Evaluate',title:'Label and evaluate',duration:7}
    ];
    let total=0;chapters.forEach(c=>{c.start=total;total+=c.duration});
    const clamp=(x,a=0,b=1)=>Math.max(a,Math.min(b,x)),mix=(a,b,t)=>a+(b-a)*t;
    const ease=t=>{t=clamp(t);return t*t*(3-2*t)};
    const span=(p,a,b)=>ease((p-a)/(b-a));
    const CUM=[0];for(let i=1;i<S.route.length;i++)CUM.push(CUM[i-1]+Math.hypot(S.route[i][0]-S.route[i-1][0],S.route[i][1]-S.route[i-1][1]));
    const chosen=S.candidates[S.chosen],kept=S.candidates.map((c,i)=>({...c,index:i})).filter(c=>c.rank).sort((a,b)=>a.rank-b.rank);
    let elapsed=0,current=null;
    const FAN_BOX=(()=>{let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;for(const f of S.fan)for(const [x,y] of f.points){x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y)}return {x0,y0,x1,y1}})();
    function at(s){s=clamp(s,0,S.s_contact);let i=1;while(i<CUM.length-1&&CUM[i]<s)i++;const p=(s-CUM[i-1])/(CUM[i]-CUM[i-1]),a=S.route[i-1],b=S.route[i];return [mix(a[0],b[0],p),mix(a[1],b[1],p)]}
    function sample(c,t){const f=clamp((t+1)/S.dt,0,c.arc.length-1),i=Math.floor(f);return mix(c.arc[i],c.arc[Math.min(i+1,c.arc.length-1)],f-i)}
    const ex=t=>-28.4+8*t;
    const TP=p=>[1150+35*p[0]-20*p[1],369-5.075*p[0]-28.5*p[1]];
    const cam=(x,y,w)=>({x,y,w});
    const mixCam=(a,b,t)=>cam(mix(a.x,b.x,t),mix(a.y,b.y,t),mix(a.w,b.w,t));
    const view={x:0,y:0,w:W,h:H};
    const Z={intro0:cam(700,393,1340),intro1:cam(1050,393,1340),contact:cam(1510,370,1070),exposure:cam(1480,388,1110),starts:cam(1625,416,780),generate:cam(1610,405,850),wide:cam(750,393,1340)};
    function stateAt(t){
      t=clamp(t,0,total-.00001);const k=Math.max(0,chapters.findIndex(c=>t<c.start+c.duration)),p=clamp((t-chapters[k].start)/chapters[k].duration);
      let time=0,camera=Z.starts;
      switch(k){
        case 0:time=mix(-1,1.6,span(p,.02,.95));camera=mixCam(Z.intro0,Z.intro1,ease(p));break;
        case 1:time=mix(1.6,5,span(p,.12,.85));camera=mixCam(Z.intro1,Z.contact,span(p,0,.65));break;
        case 2:time=mix(5,3.5,span(p,.06,.40));camera=mixCam(Z.contact,Z.exposure,span(p,0,.42));break;
        case 3:time=p<.47?mix(3.5,0,span(p,.05,.43)):mix(0,-1,span(p,.50,.89));camera=mixCam(Z.exposure,Z.starts,span(p,0,.25));break;
        case 4:time=mix(-1,3.5,span(p,.37,.94));camera=Z.starts;break;
        case 5:time=3.5;camera=Z.starts;break;
        case 6:time=0;camera=mixCam(Z.starts,Z.generate,span(p,0,.22));break;
        case 7:time=mix(-1,5,span(p,.02,.95));camera=p<.15?mixCam(Z.generate,Z.wide,span(p,0,.15)):mixCam(Z.wide,Z.contact,span(p,.15,.95));break;
        case 8:time=p<.28?3.5:p<.52?1.4:mix(1.6,5,span(p,.54,.94));camera=Z.exposure;break;
      }
      return {k,p,time,camera,elapsed:t};
    }
    function setup(c,v=view){
      const scale=v.w/c.w,h=v.h/scale;
      current={cam:c,v,k:scale,left:c.x-c.w/2,top:c.y-h/2};
    }
    function P(p){const a=TP(p),q=current;return [q.v.x+(a[0]-q.left)*q.k,q.v.y+(a[1]-q.top)*q.k]}
    const FONT='"Helvetica Neue", Helvetica, Arial, sans-serif',MATH='"Times New Roman", Times, serif';
    const runsOf=s=>(Array.isArray(s)?s:[String(s)]).map(r=>typeof r==='string'?{t:r}:r);
    const fontOf=(r,size,weight)=>{const z=r.sub||r.sup?size*.72:size;return r.italic?`italic ${weight} ${z*1.12}px ${MATH}`:`${weight} ${z}px ${FONT}`};
    function measure(s,size,weight=500){weight=Math.round(weight/100)*100;let w=0;for(const r of runsOf(s)){ctx.font=fontOf(r,size,weight);w+=ctx.measureText(r.t).width}return w}
    function text(s,x,y,size=13,color=C.ink,weight=500,align='left'){
      weight=Math.round(weight/100)*100;const w=measure(s,size,weight);let cx=align==='center'?x-w/2:align==='right'?x-w:x;
      ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillStyle=color;
      for(const r of runsOf(s)){ctx.font=fontOf(r,size,weight);ctx.fillText(r.t,cx,y+(r.sub?size*.32:r.sup?-size*.36:0));cx+=ctx.measureText(r.t).width}
    }
    const te=(rest='')=>[{t:'t',italic:true},{t:'e',sub:true,italic:true},{t:'*',sup:true},rest],tc=(rest='')=>[{t:'t',italic:true},{t:'c',sub:true,italic:true},{t:'*',sup:true},rest],wstar=(rest='')=>[{t:'W',italic:true},{t:'*',sup:true},rest];
    function round(x,y,w,h,r=12,fill='#fffdf9',stroke=null){
      ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.8;ctx.stroke()}
    }
    // a frosted-glass panel: the picture behind it blurred, a translucent tint over it and a light rim
    function glass(x,y,w,h,r=14,tint='rgba(250,248,245,.42)',rim='rgba(255,255,255,.55)'){
      const m=ctx.getTransform(),k=m.a,pad=14;
      ctx.save();ctx.shadowColor='rgba(18,34,43,.16)';ctx.shadowBlur=18;ctx.shadowOffsetY=4;ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.fillStyle='rgba(0,0,0,.001)';ctx.fill();ctx.restore();
      ctx.save();ctx.beginPath();ctx.roundRect(x,y,w,h,r);ctx.clip();
      try{ctx.filter='blur(10px) saturate(1.2)';ctx.drawImage(ctx.canvas,(x-pad)*k+m.e,(y-pad)*k+m.f,(w+2*pad)*k,(h+2*pad)*k,x-pad,y-pad,w+2*pad,h+2*pad)}catch(e){}
      ctx.filter='none';ctx.fillStyle=tint;ctx.fillRect(x,y,w,h);
      const g=ctx.createLinearGradient(0,y,0,y+h);g.addColorStop(0,'rgba(255,255,255,.18)');g.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=g;ctx.fillRect(x,y,w,h);
      ctx.restore();
      ctx.save();ctx.beginPath();ctx.roundRect(x+.5,y+.5,w-1,h-1,r);ctx.strokeStyle=rim;ctx.lineWidth=1;ctx.stroke();ctx.restore();
    }
    function card(x,y,w,h,alpha=.96,r=13){glass(x,y,w,h,r,`rgba(250,248,245,${Math.min(.62,alpha*.5)})`)}
    // a short label: white pill with a coloured dot and dark text
    function pill(s,x,y,color=C.ink,alpha=1,size=14){ctx.save();ctx.globalAlpha*=alpha;const h=size+14,w=measure(s,size,600)+33;card(x-w/2,y-h/2,w,h,.97,h/2);ctx.fillStyle=color;ctx.beginPath();ctx.arc(x-w/2+13,y,4.2,0,Math.PI*2);ctx.fill();text(s,x-w/2+23,y+.4,size,C.ink,600);ctx.restore()}
    function line(points,color,width=1.5,dash=[],alpha=1){if(points.length<2||alpha<=0)return;ctx.save();ctx.globalAlpha*=alpha;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.setLineDash(dash);ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();ctx.restore()}
    function worldLine(points,color,width=1.5,dash=[],alpha=1){line(points.map(P),color,width,dash,alpha)}
    // a screen-space line that grows stronger from its start to its end
    function softLine(pts,color,width=1,alpha=1){if(pts.length<2||alpha<=.01)return;const a=pts[0],b=pts[pts.length-1],g=ctx.createLinearGradient(a[0],a[1],b[0],b[1]);
      g.addColorStop(0,color+'22');g.addColorStop(1,color);ctx.save();ctx.globalAlpha*=alpha;ctx.strokeStyle=g;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();ctx.restore()}
    // a screen-space line with a soft glow in its own colour, optionally ending in an arrowhead
    function glowLine(pts,color,width=2,alpha=1,{arrow=false,dash=[]}={}){if(pts.length<2||alpha<=.01)return;ctx.save();ctx.globalAlpha*=alpha;ctx.shadowColor=color;ctx.shadowBlur=9;
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineCap='round';ctx.lineJoin='round';ctx.setLineDash(dash);ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();
      if(arrow){const b=pts[pts.length-1],a=pts[Math.max(0,pts.length-3)],an=Math.atan2(b[1]-a[1],b[0]-a[0]),L=width*3.4;ctx.setLineDash([]);ctx.fillStyle=color;ctx.beginPath();
        ctx.moveTo(b[0]+Math.cos(an)*L*.6,b[1]+Math.sin(an)*L*.6);ctx.lineTo(b[0]+Math.cos(an+2.5)*L,b[1]+Math.sin(an+2.5)*L);ctx.lineTo(b[0]+Math.cos(an-2.5)*L,b[1]+Math.sin(an-2.5)*L);ctx.closePath();ctx.fill()}
      ctx.restore()}
    // a small coloured disc with a white tick or cross
    function badge(x,y,good,alpha=1){ctx.save();ctx.globalAlpha*=alpha;ctx.fillStyle=good?C.green:C.red;ctx.beginPath();ctx.arc(x,y,6.5,0,Math.PI*2);ctx.fill();good?tick(x-2.5,y+.5,'#fff'):cross(x,y,'#fff');ctx.restore()}
    function ring(p,color,r=8,alpha=1){const a=P(p);ctx.save();ctx.globalAlpha*=alpha;ctx.fillStyle=color+(r>12?'14':'40');ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.beginPath();ctx.ellipse(a[0],a[1],r,r*.60,-.14,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore()}
    function tick(x,y,color=C.green){line([[x-3,y],[x,y+3],[x+6,y-5]],color,1.8)}
    function cross(x,y,color=C.red){line([[x-3,y-3],[x+3,y+3]],color,1.7);line([[x-3,y+3],[x+3,y-3]],color,1.7)}
    function pin(point,letter,color,alpha=1){const p=P(point),top=[p[0]+5,p[1]-40];ctx.save();ctx.globalAlpha*=alpha;ring(point,color,11);line([p,top],color,2);
      ctx.shadowColor='rgba(18,34,43,.30)';ctx.shadowBlur=7;ctx.shadowOffsetY=2;ctx.fillStyle=color;ctx.beginPath();ctx.arc(top[0],top[1]-5,13,0,Math.PI*2);ctx.fill();ctx.shadowColor='transparent';
      text(letter,top[0],top[1]-4.5,14,'#fff',700,'center');ctx.restore()}
    // a pill tied to a scene point by a thin leader line
    function label(point,s,color=C.ink,dx=38,dy=-22,alpha=1){if(alpha<=0)return;const a=P(point),x=a[0]+dx,y=a[1]+dy;line([a,[x,y]],'rgba(28,46,62,.55)',1,[],alpha);ctx.save();ctx.globalAlpha*=alpha;ctx.fillStyle=color;ctx.beginPath();ctx.arc(a[0],a[1],2.8,0,Math.PI*2);ctx.fill();ctx.restore();pill(s,x,y,color,alpha)}
    function background(){const q=current;ctx.save();ctx.beginPath();ctx.rect(q.v.x,q.v.y,q.v.w,q.v.h);ctx.clip();ctx.fillStyle='#d8d6cb';ctx.fillRect(q.v.x,q.v.y,q.v.w,q.v.h);ctx.translate(q.v.x,q.v.y);ctx.scale(q.k,q.k);ctx.translate(-q.left,-q.top);ctx.drawImage(images.plate,0,0);ctx.restore()}
    function shadows(time,alpha=.17){const polys=S.shadows[Math.round(clamp((time+1)/S.dt,0,S.frames-1))];ctx.save();ctx.fillStyle=`rgba(27,43,55,${alpha})`;ctx.beginPath();for(const ps of polys){ps.map(P).forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath()}ctx.fill();ctx.restore()}
    function route(time,{amount=1,split=false,width=3,alpha=.95}={}){
      const end=S.s_contact*clamp(amount),seen=S.visible[Math.round(clamp((time+1)/S.dt,0,S.frames-1))],r=Math.max(2.3,1.75*Math.sqrt(current.k)*width/3);
      ctx.save();
      for(let s=0.2;s<end;s+=0.42){
        const vis=!split||seen.some(([a,z])=>s>=a-.03&&s<=z+.03),q=P(at(s));
        ctx.globalAlpha=alpha*(vis?1:.85);ctx.fillStyle=vis?C.orange:'#6f7b82';ctx.beginPath();ctx.arc(q[0],q[1],vis?r*1.15:r*.85,0,Math.PI*2);ctx.fill();
      }
      ctx.restore();
    }
    function carImage(x,y,index=0,alpha=1,scale=1){
      const a=P([x,y]),c=A.cars.cells[index],f=(4.6*35+1.9*20)/c.w*current.k*scale;
      ctx.save();ctx.globalAlpha*=alpha;ctx.drawImage(images.cars,c.x,c.y,c.w,c.h,a[0]-(c.w/2)*f,a[1]-c.h*.62*f,c.w*f,c.h*f);ctx.restore();
    }
    function personScreen(x,y,{height=54,alpha=1,hidden=false,frame=0,direction=0,badge=null}={}){
      const n=(direction?8:0)+(Math.floor(frame)%8+8)%8,c=A.walk.cells[n],f=height/c.h;
      ctx.save();ctx.globalAlpha*=alpha*(hidden?.43:1);ctx.translate(x,y);
      if(!direction)ctx.scale(-1,1);
      ctx.drawImage(images.walk,c.x,c.y,c.w,c.h,(c.x-c.pivotX)*f,-c.h*f,c.w*f,c.h*f);ctx.restore();
      if(badge){ctx.save();ctx.globalAlpha*=alpha;ctx.fillStyle=badge==='bad'?C.red:C.green;ctx.beginPath();ctx.arc(x+10,y-height-1,6.5,0,Math.PI*2);ctx.fill();badge==='bad'?cross(x+10,y-height-1,'#fff'):tick(x+10,y-height-1,'#fff');ctx.restore()}
    }
    function personImage(s,time,{point=null,alpha=1,hidden=false,scale=1,badge=null,standing=false,go=0}={}){
      const a=P(point||at(s)),frame=standing&&time<go?2:Math.floor((time-go+20)*7)%8;
      personScreen(a[0],a[1],{height:54*current.k*scale,alpha,hidden,frame,direction:s>=19.2?1:0,badge});
    }
    function parked(alpha=1){const entries=Object.values(S.occluders).map((b,i)=>({b,i,y:P([(b[0]+b[2])/2,(b[1]+b[3])/2])[1]})).sort((a,b)=>a.y-b.y);for(const {b,i}of entries)carImage((b[0]+b[2])/2,(b[1]+b[3])/2,i+1,alpha)}
    function history(c,alpha=1){const points=[];for(let t=-1;t<=0;t+=.1)points.push(at(sample(c,t)));worldLine(points,C.orange,2,[2,5],alpha*.6);for(let t=-1;t<=.001;t+=.25){const p=P(at(sample(c,t)));ctx.save();ctx.globalAlpha*=alpha*.7;ctx.fillStyle=C.orange;ctx.beginPath();ctx.arc(p[0],p[1],2.3,0,Math.PI*2);ctx.fill();ctx.restore()}}
    function candidates(time,k,p){
      const entries=[];
      S.candidates.forEach((c,i)=>{
        const bad=c.drop==='leaves_before_start'||c.drop==='seen_early',early=c.drop==='leaves_before_start',showBad=k===4&&bad;
        const fade=showBad?1-span(p,early?.08:.25,early?.23:.39):1;
        if(fade<.002)return;
        let s=sample(c,time),pt=at(s),merge=span(time,2.25,3.5),side=c.kind==='moving'?-.60:((i-4)%3-1)*.28;
        const d=s<19.2?[1,0]:s<20.614?[Math.SQRT1_2,Math.SQRT1_2]:[0,1];
        pt=[pt[0]-d[1]*side*(1-merge),pt[1]+d[0]*side*(1-merge)];
        entries.push({i,c,s,pt,fade,showBad,y:P(pt)[1]});
      });
      entries.sort((a,b)=>a.y-b.y);
      for(const e of entries)personImage(e.s,time,{point:e.pt,alpha:e.fade*(e.i===S.chosen?.95:.48),hidden:!(k===4&&p>.91),scale:e.i===S.chosen?1:.83,badge:e.showBad?'bad':k===4&&p>.91?'ok':null,standing:e.c.kind==='standing',go:e.c.t_go||0});
    }
    function scene(camera,time,options={},extra=()=>{},v=view){
      setup(camera,v);ctx.save();ctx.beginPath();ctx.rect(v.x,v.y,v.w,v.h);ctx.clip();background();
      if(options.shadow)shadows(time,options.shadowAlpha||.17);
      if(options.route)route(time,options.route===true?{}:options.route);
      if(options.history)history(chosen,options.history);
      if(options.fan)options.fan();
      parked(options.parkAlpha===undefined?1:options.parkAlpha);
      if(options.ego!==false&&options.corridor!==false)corridor(options.egoX===undefined?ex(time):options.egoX);
      if(options.ego!==false)carImage(options.egoX===undefined?ex(time):options.egoX,0,0);
      if(options.candidates)candidates(time,options.candidates.k,options.candidates.p);
      if(options.ped){const s=sample(chosen,time);personImage(s,time,{hidden:time<S.t_e,alpha:1})}
      extra();ctx.restore();
    }
    // the lane ahead of the reference ego: a blue band as wide as the car that fades with distance
    function corridor(x,length=30,half=1.05){
      const a=P([x,0]),b=P([x+length,0]),g=ctx.createLinearGradient(a[0],a[1],b[0],b[1]);g.addColorStop(0,'rgba(63,130,193,.40)');g.addColorStop(.6,'rgba(63,130,193,.16)');g.addColorStop(1,'rgba(63,130,193,0)');
      const e=ctx.createLinearGradient(a[0],a[1],b[0],b[1]);e.addColorStop(0,'rgba(63,130,193,.95)');e.addColorStop(1,'rgba(63,130,193,0)');
      ctx.save();ctx.fillStyle=g;ctx.beginPath();[[x-1.5,-half],[x+length,-half],[x+length,half],[x-1.5,half]].map(P).forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();
      ctx.strokeStyle=e;ctx.lineWidth=1.6*Math.sqrt(current.k);for(const y of [-half,half]){ctx.beginPath();ctx.moveTo(...P([x-1.5,y]));ctx.lineTo(...P([x+length,y]));ctx.stroke()}
      ctx.restore();
    }
    // the whole street seen from further back, drawn with the same picture, cars and shadows
    function insetView(time){
      const r={x:22,y:372,w:306,h:118},saved=current;
      glass(r.x-5,r.y-5,r.w+10,r.h+10,15,'rgba(22,33,44,.50)','rgba(255,255,255,.28)');
      scene(cam(1000,400,1660),time,{shadow:true,route:{split:true,width:2.4}},()=>{},r);
      ctx.save();ctx.beginPath();ctx.rect(r.x,r.y,r.w,r.h);ctx.clip();
      const g=ctx.createLinearGradient(0,r.y,0,r.y+30);g.addColorStop(0,'rgba(12,20,28,.55)');g.addColorStop(1,'rgba(12,20,28,0)');ctx.fillStyle=g;ctx.fillRect(r.x,r.y,r.w,30);
      text('reference ego',r.x+10,r.y+13,13,'#cfe3fa',700);text(`${time<0?'−':''}${Math.abs(time).toFixed(1)} s`,r.x+r.w-10,r.y+13,13,'#ffffff',600,'right');
      const e=P([ex(time),0]);if(e[0]<r.x+6){const yy=clamp(e[1],r.y+40,r.y+r.h-12);ctx.fillStyle=C.blue;ctx.beginPath();ctx.moveTo(r.x+6,yy);ctx.lineTo(r.x+17,yy-7);ctx.lineTo(r.x+17,yy+7);ctx.closePath();ctx.fill()}
      ctx.restore();current=saved;
    }
    function miniMap(time){
      const x=22,y=372,w=270,h=112,f=Math.round(clamp((time+1)/S.dt,0,S.frames-1));
      glass(x,y,w,h,14,'rgba(22,33,44,.58)','rgba(255,255,255,.22)');
      text('reference ego',x+15,y+15,10.5,'#a9cdf5',600);text(`${time<0?'−':''}${Math.abs(time).toFixed(1)} s`,x+w-15,y+15,10.5,'#eef2f5',500,'right');
      const M=p=>[x+14+(p[0]+39)*3.85,y+67-p[1]*3.85];
      ctx.save();ctx.beginPath();ctx.roundRect(x+9,y+27,w-18,h-35,9);ctx.clip();
      const band=(y0,y1,color)=>{const a=M([-45,y1]),b=M([35,y0]);ctx.fillStyle=color;ctx.fillRect(a[0],a[1],b[0]-a[0],b[1]-a[1])};
      band(5.25,7.6,'rgba(190,196,190,.20)');band(-4.6,5.25,'rgba(96,106,116,.70)');band(-6.8,-4.6,'rgba(190,196,190,.24)');
      line([M([-45,1.75]),M([35,1.75])],'rgba(255,255,255,.55)',1,[5,5]);line([M([-45,-1.75]),M([35,-1.75])],'rgba(255,255,255,.30)',.8);
      const e=M([ex(time),0]),g=ctx.createRadialGradient(e[0],e[1],0,e[0],e[1],95);g.addColorStop(0,'rgba(255,214,130,.38)');g.addColorStop(1,'rgba(255,214,130,0)');
      ctx.fillStyle=g;ctx.beginPath();ctx.moveTo(...e);ctx.arc(e[0],e[1],95,-1.25,1.25);ctx.closePath();ctx.fill();
      ctx.fillStyle='rgba(6,12,18,.42)';ctx.beginPath();for(const ps of S.shadows[f]){ps.forEach((p,i)=>{const q=M(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.closePath()}ctx.fill();
      ctx.fillStyle=C.orange;for(let s=.2;s<S.s_contact;s+=.9){const q=M(at(s));ctx.beginPath();ctx.arc(q[0],q[1],1.1,0,Math.PI*2);ctx.fill()}
      const car=([x0,y0,x1,y1],color)=>{const a=M([x0,y1]),b=M([x1,y0]),cw=b[0]-a[0],ch=b[1]-a[1];round(a[0],a[1],cw,ch,2.6,color);ctx.fillStyle='rgba(12,20,28,.55)';ctx.fillRect(a[0]+cw*.56,a[1]+1.1,cw*.15,ch-2.2);ctx.fillRect(a[0]+cw*.18,a[1]+1.4,cw*.1,ch-2.8)};
      Object.values(S.occluders).forEach(b=>car(b,'#aeb6bd'));
      car([ex(time)-2.3,-.95,ex(time)+2.3,.95],'#4f93e0');
      for(const [pt,color] of [[S.contact,C.red],[S.exposure,C.green]]){const q=M(pt);ctx.strokeStyle=color;ctx.lineWidth=1.4;ctx.beginPath();ctx.arc(q[0],q[1],3.2,0,Math.PI*2);ctx.stroke()}
      ctx.restore();
    }
    function roster(k,p,time){
      card(22,100,306,262,.96);
      text(k===4&&p>.90?'9 hidden starts':'19 candidate starts',40,124,18,C.ink,700);
      text('moving',40,152,13,'#5b6870',600);
      for(let i=0;i<4;i++){
        const c=S.candidates[i],x=80+i*66,bad=k===4&&i===3&&p>.25,alpha=bad?1-span(p,.29,.4)*.8:1;
        personScreen(x,208,{height:46,alpha,frame:2,direction:0,badge:bad?'bad':k===4&&p>.91?'ok':null});
        text(`${c.v0.toFixed(2)} m/s`,x,226,12,bad?C.red:'#4f5c64',600,'center');
      }
      text('standing',40,252,13,'#5b6870',600);
      for(let b=0;b<5;b++){
        const x=58+b*58;personScreen(x,302,{height:38,frame:2,direction:1});
        for(let a=0;a<3;a++){
          const c=S.candidates[4+b*3+a],bad=k===4&&c.drop==='leaves_before_start'&&p>.08,good=k===4&&p>.91&&c.rank;
          ctx.fillStyle=bad?C.red:good?C.green:C.orange;ctx.globalAlpha=bad?.3:.9;ctx.beginPath();ctx.arc(x-9+a*9,315,2.6,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
          if(bad)cross(x-9+a*9,315,C.red);
        }
      }
      let s=k===3?(p<.47?'solve initial states':'1 s history'):p<.24?'leaves before start':p<.40?'seen early':p<.91?'hidden until exposure':'9 reach E unseen';
      text(s,175,343,14,k===4&&p<.40?C.red:k===4&&p>.91?C.green:C.orange,700,'center');
    }
    function header(st){
      const title=chapters[st.k].title;glass(18,16,measure(title,19,600)+70,44,22,'rgba(250,248,245,.50)');
      {const g=ctx.createLinearGradient(0,25,0,48);g.addColorStop(0,'#f07a4f');g.addColorStop(1,'#d9471f');ctx.fillStyle=g;ctx.beginPath();ctx.arc(41,38,13,0,Math.PI*2);ctx.fill()}
      text(String(st.k+1),41,38.6,14,'#fff',700,'center');text(title,63,38.6,19,C.ink,600);
      const x0=612,x1=956,y=50,X=t=>x0+(t+1)/7*(x1-x0);
      glass(588,16,390,st.k>=2?88:70,16,'rgba(250,248,245,.50)');
      if(st.k>=3){round(X(-1),y-5,X(0)-X(-1),10,5,'#efe2d2');text('history',(X(-1)+X(0))/2,y+31,12.5,'#8a7360',600,'center')}
      if(st.k>=2){const q=st.k===2?span(st.p,.04,.40):1,xx=mix(X(5),X(3.5),q);round(xx,y-5,X(5)-xx,10,5,'rgba(46,150,113,.30)');text(wstar(' = 1.5 s'),(X(3.5)+X(5))/2,y+31,14,C.green,600,'center')}
      line([[x0,y],[x1,y]],'#b8c0c2',1.2);
      for(let i=-1;i<=6;i++){line([[X(i),y-2],[X(i),y+3]],'#a3acae',.9);text(i<0?'−1':`${i}${i===6?' s':''}`,X(i),y+15,12,'#6f7c82',500,'center')}
      if(st.k>=1){line([[X(5),y-9],[X(5),y+8]],C.red,2);text(tc(),X(5),y-19,17,C.red,600,'center')}
      if(st.k>=2){line([[X(3.5),y-9],[X(3.5),y+8]],C.green,2);text(te(),X(3.5),y-19,17,C.green,600,'center')}
      const xp=X(st.time);ctx.fillStyle=C.ink;ctx.beginPath();ctx.arc(xp,y,4.2,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fffdf9';ctx.lineWidth=1.6;ctx.stroke();
      ctx.fillStyle='rgba(250,248,245,.45)';ctx.fillRect(0,H-2,W,2);ctx.fillStyle=C.orange;ctx.fillRect(0,H-2,W*st.elapsed/total,2);
    }
    function endpointPins(alphaE=1,alphaC=1){pin(S.contact,'C',C.red,alphaC);pin(S.exposure,'E',C.green,alphaE)}
    function phase0(st){scene(st.camera,st.time,{},()=>{label([ex(st.time),0],'reference ego',C.blue,10,-82,span(st.p,.10,.25))})}
    function phase1(st){scene(st.camera,st.time,{route:{amount:span(st.p,.04,.44)}},()=>{
      pin(S.contact,'C',C.red,span(st.p,.25,.45));
      label(S.contact,st.p>.77?'C at 5.0 s':'contact C',C.red,34,-76,span(st.p,.30,.48));
      const r=at(18);if(st.p<.65)label(r,'target route',C.orange,-65,35,span(st.p,.10,.30)*(1-span(st.p,.45,.65)));
    })}
    function phase2(st){scene(st.camera,st.time,{shadow:st.p>.35,route:{split:st.p>.48}},()=>{
      endpointPins(span(st.p,.62,.78),1);
      label(S.contact,'C at 5.0 s',C.red,-70,-52);
      if(st.p>.5){const a=P([ex(st.time),0]),b=P(S.exposure);glowLine([[a[0],a[1]-20],b],C.green,1.6,span(st.p,.5,.7));
        label(S.exposure,'E at 3.5 s',C.green,75,-12,span(st.p,.65,.79));
        label(at(20.4),'hidden',C.gray,-82,28,span(st.p,.50,.67));
        const s=mix(S.s_contact,S.s_reveal,span(st.p,.52,.80)),pt=P(at(s));ctx.fillStyle='rgba(46,150,113,.20)';ctx.beginPath();ctx.arc(pt[0],pt[1],13,0,Math.PI*2);ctx.fill();
      }
    })}
    function phase3(st){scene(st.camera,st.time,{shadow:true,route:{split:true},candidates:{k:3,p:st.p},history:st.p>.5?span(st.p,.50,.8):0},()=>{
      endpointPins();
      if(st.p>.5){label(at(sample(chosen,-1)),'−1 s',C.orange,-20,27,span(st.p,.5,.68));label(at(sample(chosen,0)),'0 s',C.orange,24,28,span(st.p,.5,.68));}
      const pt=P(at(S.s_stand));pill('standing',pt[0]+54,pt[1]+15,C.orange,span(st.p,.38,.55));
    });roster(3,st.p,st.time);insetView(st.time)}
    function phase4(st){scene(st.camera,st.time,{shadow:true,route:{split:true},candidates:{k:4,p:st.p}},()=>{
      endpointPins();
      if(st.p>.25&&st.p<.40){const c=S.candidates.find(c=>c.drop==='seen_early'),s=sample(c,-1);glowLine([P([ex(-1),0]),P(at(s))],C.red,1.6,1-span(st.p,.32,.40),{dash:[5,5]});}
      if(st.p>.90){const e=P(S.exposure);pill('9 retained',e[0]+66,e[1]+23,C.green,span(st.p,.90,.95))}
    });roster(4,st.p,st.time);insetView(st.time)}
    function phase5(st){scene(st.camera,3.5,{shadow:true,route:{split:true},ego:false});ctx.save();ctx.globalAlpha=span(st.p,0,.2);glass(0,0,W,H,0,'rgba(250,248,245,.30)','rgba(0,0,0,0)');ctx.restore();
      const q=span(st.p,.02,.35),cut=span(st.p,.64,.89);const origin=P(S.exposure);
      kept.forEach((c,i)=>{const x=mix(origin[0],62+i*109,q),y=mix(origin[1],315,q),alpha=i<5?1:1-cut*.94;
        ctx.save();ctx.globalAlpha*=alpha;card(x-43,y-80,86,126,.96);personScreen(x,y+9,{height:75,frame:2,direction:c.kind==='standing'?1:0});text(`rank ${i+1}`,x,y+33,13.5,i<5?C.green:C.gray,600,'center');if(i<5&&q>.95)tick(x+28,y-60);ctx.restore();
      });if(st.p>.4)pill('1.37 m/s',62,391,C.green,span(st.p,.4,.55));pill('pedestrian plausibility',500,170,C.ink,q);if(st.p>.65)pill('5 retained starts',500,449,C.green,span(st.p,.65,.85));
    }
    function fan(st){
      const grow=span(st.p,.25,.54),judge=span(st.p,.56,.78),pick=span(st.p,.77,.95);
      if(grow<=0)return;
      for(const f of S.fan){if(f.chosen)continue;const n=Math.max(2,Math.round(f.points.length*grow)),pts=f.points.slice(0,n).map(P);
        softLine(pts,f.ok?C.orange:'#4e5b63',f.ok?1.1:.9,(f.ok?.55:.5*(1-judge))*(1-pick*.65))}
      if(pick>0){const f=S.fan.find(f=>f.chosen);glowLine(f.points.slice(0,Math.max(2,Math.round(f.points.length*pick))).map(P),C.orange,3,pick,{arrow:pick>.95})}
    }
    function phase6(st){scene(st.camera,0,{shadow:true,ego:false,fan:()=>fan(st)},()=>{
      endpointPins();const s=sample(chosen,0);personImage(s,0,{hidden:true});if(st.p>.8)label(S.exposure,'selected',C.orange,64,11,span(st.p,.8,.94));
    });
      generatorPanel(st);
    }
    function generatorPanel(st){
      const x=22,y=104,w=322,h=336,q=span(st.p,.03,.25),grow=span(st.p,.25,.54),judge=span(st.p,.56,.78),pick=span(st.p,.77,.95);
      glass(x,y,w,h,16);
      text('trajectory generator',x+16,y+24,17,C.ink,700);text('conditioned on',x+16,y+48,12.5,'#5b6870',600);
      [['rank 1 start',C.orange],['E at 3.5 s',C.green],['C at 5.0 s',C.red]].forEach(([s,c],i)=>{const cx=x+16+[0,104,196][i],cy=y+72;ctx.save();ctx.globalAlpha*=.35+.65*q;
        const tw=measure(s,11.5,600);round(cx,cy-11,tw+24,22,11,'rgba(255,255,255,.55)');ctx.fillStyle=c;ctx.beginPath();ctx.arc(cx+10.5,cy,3.6,0,Math.PI*2);ctx.fill();text(s,cx+18,cy+.3,11.5,C.ink,600);ctx.restore()});
      const bx=x+16,by=y+96,bw=w-32,bh=120;round(bx,by,bw,bh,10,'rgba(22,33,44,.62)');
      const B=FAN_BOX,sc=Math.min((bw-24)/(B.x1-B.x0),(bh-20)/(B.y1-B.y0)),ox=bx+bw/2-(B.x0+B.x1)/2*sc,oy=by+bh/2+(B.y0+B.y1)/2*sc;
      const F=([px,py])=>[ox+px*sc,oy-py*sc];
      ctx.save();ctx.beginPath();ctx.roundRect(bx,by,bw,bh,10);ctx.clip();
      for(const f of S.fan){if(f.chosen)continue;const n=Math.max(2,Math.round(f.points.length*grow));if(grow<=0)break;
        line(f.points.slice(0,n).map(F),f.ok?'#ff9b6e':'#9aa6ad',1,[],(f.ok?.55:.45*(1-judge*.85))*(1-pick*.55))}
      if(pick>0){const f=S.fan.find(f=>f.chosen);line(f.points.slice(0,Math.max(2,Math.round(f.points.length*pick))).map(F),C.orange,2.6,[],pick)}
      ctx.restore();
      const pass=S.fan.filter(f=>f.ok).length;
      text(pick>.5?'1 selected':judge>.5?`${pass} of 64 pass`:'64 candidates',bx+12,by+bh-13,12.5,'#eef2f5',600);
      ['hidden approach','exposure confirmed','window ±0.2 s','motion limits'].forEach((s,i)=>{
        const cx=x+16+(i%2)*150,cy=y+244+Math.floor(i/2)*32,done=judge>.15+i*.18;
        ctx.save();ctx.fillStyle=done?C.green:'rgba(28,46,62,.14)';ctx.beginPath();ctx.arc(cx+9,cy,9,0,Math.PI*2);ctx.fill();if(done)tick(cx+6.5,cy+.5,'#fff');ctx.restore();
        text(s,cx+24,cy+.5,12.5,done?C.ink:'#5b6870',600)});
      text('checked in score order',x+16,y+h-18,11.5,'#5f6c74',500);
    }
    function phase7(st){scene(st.camera,st.time,{shadow:true,route:{split:true,width:3.3},history:.7,ped:true},()=>{
      endpointPins(.9,.9);const s=sample(chosen,st.time),pt=at(s),a=P([ex(st.time),0]),b=P(pt),visible=st.time>=3.5;
      glowLine([[a[0],a[1]-20],[b[0],b[1]-30]],visible?C.green:'#5c6a72',visible?1.8:1.2,visible?.95:.7,{dash:visible?[]:[4,5]});
      label(pt,visible?'visible':'hidden',visible?C.green:C.gray,48,-48);
      if(st.time>4.94){ring(S.contact,C.red,17);pill([{t:'W',italic:true},' = ',...wstar()],816,474,C.green)}
    })}
    function avoidance(st,early){const from=early?1.4:3.5,start=[ex(from),0],end=S.contact;for(let i=0;i<13;i++){
      const good=early&&i<5,pts=[];for(let j=0;j<24;j++){const u=j/23;pts.push([mix(start[0],good?8.2:end[0],u),mix(0,good?2.5:end[1],u)+(i-6)*.035*Math.sin(u*Math.PI)])}
      softLine(pts.map(P),good?C.green:C.red,1.3,.75);const q=P(pts[pts.length-1]);badge(q[0],q[1],good,.95);
    }}
    function phase8(st){
      if(st.p<.52){const early=st.p>=.28;scene(st.camera,st.time,{shadow:!early,parkAlpha:early?.17:1,route:{split:!early},ped:true},()=>{endpointPins();avoidance(st,early);pill(early?'earlier visibility':'after exposure',385,471,early?C.green:C.red);if(st.p>.40)pill('Anticipatory',756,128,C.green)});
      }else{
        scene(st.camera,3.5,{ego:false,parkAlpha:.5});glass(0,0,W,H,0,'rgba(250,248,245,.34)','rgba(0,0,0,0)');
        const sim=st.time,advance=Math.max(0,sim-2),earlyX=sim<=2?ex(sim):ex(2)+8*advance-1.5*advance*advance;
        const panels=[{x:18,y:151,w:473,h:302,open:false},{x:509,y:151,w:473,h:302,open:true}];
        for(const v of panels){
          const lateX=Math.min(ex(sim),11.6),cx=v.open?earlyX:lateX;
          scene(cam(1515,385,1050),sim,{shadow:!v.open,parkAlpha:v.open?.16:1,egoX:cx,route:true,ped:true},()=>{
            pill(v.open?'unoccluded':'masked',v.x+v.w/2,v.y+22,v.open?C.green:C.ink);
            if(st.p>.94){if(!v.open)ring(S.contact,C.red,11);pill(v.open?'avoids contact':'contact',v.x+v.w/2,v.y+v.h-22,v.open?C.green:C.red)}
          },v);
        }
        pill('Anticipatory',500,116,C.green);text('Illustrative',974,481,12.5,'#4f5c64',600,'right');
      }
      if(st.p<.52)text('Illustrative',975,534,12.5,'#fffdf9',600,'right');
    }
    const phases=[phase0,phase1,phase2,phase3,phase4,phase5,phase6,phase7,phase8];
    const stills=[[.75],[.85],[.84],[.4,.9],[.16,.31,.95],[.9],[.95],[.25,.73,.98],[.18,.46,.98]];
    function draw(t,{reduced=false}={}){
      elapsed=clamp(t,0,total-.00001);let st=stateAt(elapsed);
      if(reduced){const f=stills[st.k],p=f[Math.min(f.length-1,Math.floor(st.p*f.length))];st=stateAt(chapters[st.k].start+p*chapters[st.k].duration)}
      ctx.save();ctx.clearRect(0,0,W,H);ctx.fillStyle=C.paper;ctx.fillRect(0,0,W,H);phases[st.k](st);header(st);ctx.restore();return st;
    }
    return {draw,stateAt,total,chapters,sample,at,project:TP,kept,scene:S,width:W,height:H};
  };
});
