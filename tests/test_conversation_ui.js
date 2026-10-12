// 使用 Node 内置模块和最小 DOM 桩验证真实页面脚本的并发状态；不请求云端。
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
class Element {
    constructor() { this.childNodes = []; this.value = ''; this.style = {}; this.disabled = false; this.className = ''; this.text = ''; this.html = ''; this.classes = new Set();
        this.classList = {add:n=>this.classes.add(n), remove:n=>this.classes.delete(n), contains:n=>this.classes.has(n),
            toggle:(n,v)=>{ if(v === undefined) v=!this.classes.has(n); v?this.classes.add(n):this.classes.delete(n); }};
    }
    appendChild(node) { node.parent = this; this.childNodes.push(node); return node; }
    replaceChildren(...nodes) { this.childNodes = nodes; nodes.forEach(n=>n.parent=this); }
    remove() { if(this.parent) this.parent.childNodes = this.parent.childNodes.filter(n=>n!==this); }
    addEventListener() {} focus() {} dispatchEvent() {}
    get children() { return this.childNodes; }
    set innerHTML(s) { this.html=s; this.childNodes=[]; }
    get innerHTML() { return this.html; }
    set textContent(s) { this.text=s; }
    get textContent() { return this.text || this.childNodes.map(n=>n.textContent).join(''); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) {
        const nodes = this.childNodes.flatMap(n=>[n,...n.querySelectorAll('*')]);
        if(selector === '*') return nodes;
        if(selector === '.message.user .bubble') return nodes.filter(n=>n.className==='bubble' && n.parent?.className==='message user');
        if(selector.startsWith('.')) return nodes.filter(n=>n.className.split(' ').includes(selector.slice(1)));
        return [];
    }
}
const elements = new Map();
const element = id => {if(!elements.has(id)) elements.set(id,new Element()); return elements.get(id);};
const requests = new Map();
const pendingReads = new Map();
const storage = {getItem:()=>null,setItem:()=>{}};
const context = vm.createContext({console,AbortController,TextDecoder,Event:class{},setTimeout:()=>0,clearTimeout:()=>{},
    window:{innerWidth:1200},localStorage:storage,sessionStorage:storage,navigator:{},
    document:{documentElement:{dataset:{}},getElementById:id=>id==='welcome'?null:element(id),
        createElement:()=>new Element(),addEventListener:()=>{},querySelectorAll:()=>[]},
    fetch:async (url,options={})=>{
        if(url==='/api/chat/stream') {
            const body=JSON.parse(options.body); requests.set(body.session_id,body);
            return {ok:true,body:{getReader:()=>({read:()=>new Promise((resolve,reject)=>{
                pendingReads.set(body.session_id,resolve);
                options.signal.addEventListener('abort',()=>reject(Object.assign(new Error('stop'),{name:'AbortError'})),{once:true});
            })})}};
        }
        if(url==='/api/conversations') return {ok:true,json:async()=>({conversations:[...requests.keys()].map(session_id=>({session_id,title:session_id}))})};
        if(url.endsWith('/messages')) return {ok:true,json:async()=>({title:'B',messages:[{role:'user',content:'B 的历史'},{role:'assistant',content:'历史回复'}]})};
        if(url.endsWith('/workflow')) return {ok:true,json:async()=>({})};
        throw new Error('Unexpected request: '+url);
    }
});
const source=fs.readFileSync(path.join(__dirname,'../web/templates/index.html'),'utf8').split('<script>')[1].split('</script>')[0];
vm.runInContext(source.split('// ---------- 初始化 ----------')[0],context);
const run = code=>vm.runInContext(code,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function feed(sid,event,text) {
    const resolve=pendingReads.get(sid); assert(resolve,'流式请求应等待下一个事件'); pendingReads.delete(sid);
    resolve({done:false,value:Buffer.from(`event: ${event}\ndata: ${JSON.stringify({text})}\n\n`)}); await tick();
}
(async()=>{
    run("currentSessionId='A';userInput.value='A 长问题'");
    const a=run('sendMessage()'); await tick();
    assert.equal(element('send-btn').textContent,'■');
    await run("selectConversation('B')");
    assert.equal(element('send-btn').textContent,'➤');
    assert(element('chat-inner').textContent.includes('B 的历史'));
    assert(element('conv-list').innerHTML.includes('conv-spinner'));
    run("userInput.value='B 新问题'"); const b=run('sendMessage()'); await tick();
    assert.equal(run('sessionStreams.size'),2);
    await feed('A','token','A 的后台回复');
    assert(!element('chat-inner').textContent.includes('A 的后台回复'));
    await feed('B','token','B 的回复'); await feed('B','done','');
    pendingReads.get('B')({done:true}); await b;
    assert.equal(element('send-btn').textContent,'➤');
    await run("selectConversation('A')");
    assert(element('chat-inner').textContent.includes('A 的后台回复'));
    assert.equal(element('send-btn').textContent,'■');
    await feed('A','answer','A 的最终完整回复'); await feed('A','done','');
    pendingReads.get('A')({done:true}); await a; await tick();
    assert.equal(element('send-btn').textContent,'➤');
    assert(!element('conv-list').innerHTML.includes('conv-spinner'));
    await run("selectConversation('B');"); await run("selectConversation('A');");
    assert(element('chat-inner').querySelectorAll('.bubble').some(n=>n.innerHTML.includes('A 的最终完整回复')));
    assert.equal(requests.get('A').query,'A 长问题');
    assert.equal(requests.get('B').query,'B 新问题');
    run("userInput.value='A 再次回复'");const stopped=run('sendMessage()'); await tick();
    await run("selectConversation('B')");
    assert.equal(element('send-btn').textContent,'➤');
    await run("selectConversation('A')");await run('sendMessage()');await stopped;
    assert.equal(run('sessionStreams.size'),0);
    assert(element('chat-inner').textContent.includes('已停止'));
    console.log('PASS: 双会话并行、后台流式隔离、历史切换、完整结果保留、侧栏动画、当前会话停止');
})().catch(error=>{console.error(error);process.exitCode=1;});
