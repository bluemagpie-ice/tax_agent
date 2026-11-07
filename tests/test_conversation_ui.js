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
    addEventListener() {} focus() {} dispatchEvent() {} setAttribute() {} removeAttribute() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
    add(option) { this.appendChild(option); }
    get children() { return this.childNodes; }
    set value(value) { this.inputValue = String(value); }
    get value() { return this.inputValue; }
    get previousElementSibling() { const siblings=this.parent?.childNodes || []; return siblings[siblings.indexOf(this)-1]; }
    set innerHTML(s) { this.html=s; this.childNodes=[]; }
    get innerHTML() { return this.html; }
    set textContent(s) { this.text=s; }
    get textContent() { return this.text || this.childNodes.map(n=>n.textContent).join(''); }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    querySelectorAll(selector) {
        const nodes = this.childNodes.flatMap(n=>[n,...n.querySelectorAll('*')]);
        if(selector === '*') return nodes;
        if(selector === '.message.user .bubble') return nodes.filter(n=>n.className==='bubble' && n.parent?.className==='message user');
        if(selector === '.message.assistant') return nodes.filter(n=>n.className==='message assistant');
        if(selector.startsWith('.')) return nodes.filter(n=>n.className.split(' ').includes(selector.slice(1)));
        return [];
    }
}
const elements = new Map();
const element = id => {if(!elements.has(id)) elements.set(id,new Element()); return elements.get(id);};
const requests = new Map();
const pendingReads = new Map();
function makeStorage() {
    const data = new Map();
    return {getItem:key=>data.get(key) ?? null,setItem:(key,value)=>data.set(key,value),removeItem:key=>data.delete(key)};
}
const listeners = new Map();
const window = {innerWidth:1200,addEventListener:(type,fn)=>listeners.set(type,fn),
    dispatchEvent:event=>listeners.get(event.type)?.(event)};
const calls = [];
const providers = {GLM:'https://glm.example/v1/', Gemini:'https://gemini.example/v1/'};
let llm = {provider:'GLM',base_url:providers.GLM,model:'glm',api_key_set:true};
const settingsWrites = [], testedModels = [], personalModels = new Map();
let failSave = false;
const response = (data,status=200) => ({ok:status===200,status,json:async()=>data});
const context = vm.createContext({console,AbortController,AbortSignal,Headers,DOMException,TextDecoder,Event:class{},
    CustomEvent:class { constructor(type,{detail}) { this.type=type; this.detail=detail; } },
    Option:class { constructor(text,value) { this.textContent=text; this.value=value; } },
    setTimeout:()=>0,clearTimeout:()=>{},
    window,localStorage:makeStorage(),sessionStorage:makeStorage(),navigator:{},
    document:{documentElement:{dataset:{}},getElementById:id=>id==='welcome'?null:element(id),
        createElement:()=>new Element(),addEventListener:()=>{},querySelectorAll:()=>[]},
    fetch:async (url,options={})=>{
        const token = new Headers(options.headers).get('Authorization');
        calls.push([url,token]);
        if(url==='/api/auth/login') {
            const {credential} = JSON.parse(options.body);
            if(credential==='login-user-fixture') return response({access_token:'user-fixture'});
            if(credential==='login-admin-fixture') return response({access_token:'admin-fixture'});
            return response({detail:'登录凭证无效'},401);
        }
        if(url==='/api/me') {
            if(token==='Bearer user-fixture') return response({id:'alice',role:'user'});
            if(token==='Bearer admin-fixture') return response({id:'bob',role:'admin'});
            return response({detail:'访问 Token 无效或已失效，请重新登录'},401);
        }
        if(url==='/api/settings') {
            assert.equal(token,'Bearer admin-fixture');
            if(options.method==='POST') {
                const {llm:changes} = JSON.parse(options.body);
                settingsWrites.push(changes);
                const {api_key, ...fields} = changes;
                llm = {...llm,...fields,api_key_set:!!api_key || llm.api_key_set};
                return response({success:true});
            }
            return response({config:{llm,qdrant:{url:'',vector_limit:0}},providers});
        }
        if(url==='/api/settings/personal') {
            const saved = personalModels.get(token) || {model:'',base_url:'',provider:'自定义',api_key_set:false};
            if(options.method==='PUT') {
                const {api_key, ...fields} = JSON.parse(options.body);
                if(failSave) return response({detail:'模型配置保存失败'},500);
                settingsWrites.push({api_key,...fields});
                personalModels.set(token,{...saved,...fields,api_key_set:!!api_key || saved.api_key_set});
                return response({success:true});
            }
            return response({config:{llm:saved},providers});
        }
        if(url==='/api/settings/personal/connection') {
            testedModels.push(JSON.parse(options.body));
            return response({success:false,message:'API Key 无效'});
        }
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
const authSource=fs.readFileSync(path.join(__dirname,'../web/static/auth.js'),'utf8');
vm.runInContext(authSource,context);
vm.runInContext(source,context);
const run = code=>vm.runInContext(code,context);
const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function feed(sid,event,text) {
    const resolve=pendingReads.get(sid); assert(resolve,'流式请求应等待下一个事件'); pendingReads.delete(sid);
    resolve({done:false,value:Buffer.from(`event: ${event}\ndata: ${JSON.stringify({text})}\n\n`)}); await tick();
}
(async()=>{
    await tick();
    assert.equal(calls.length,0);
    assert.equal(element('auth-modal').open,true);
    element('auth-token').value='bad-token';
    await run('applyAccessToken()');
    assert.match(element('auth-status').textContent,/登录凭证无效/);
    assert.deepEqual(calls.map(([url])=>url),['/api/auth/login']);
    element('auth-token').value='login-user-fixture';
    await run('applyAccessToken()');
    await tick();
    assert.equal(element('role-label').textContent,'普通用户');
    assert.equal(element('auth-modal').open,false);
    assert(calls.some(([url,token])=>url==='/api/conversations' && token==='Bearer user-fixture'));
    assert(!calls.some(([url])=>url==='/api/settings'));
    assert.equal(context.sessionStorage.getItem('tax_agent_user_session'),'user-fixture');
    await run('showSettings()');
    assert.equal(element('settings-model-config').hidden,false);
    assert.equal(element('settings-scope-switch').hidden,true);
    element('cfg-llm-provider').value='GLM';
    run("selectProvider('llm')");
    element('cfg-llm-model').value='personal-model';
    element('cfg-llm-key').value='personal-fixture-key';
    await run('saveSettings()');
    assert.match(element('settings-save-status').textContent,/个人配置已保存/);
    element('cfg-llm-key').value='replacement-model-key';
    await run("testChatConnection(document.getElementById('settings-test'))");
    assert.equal(testedModels.at(-1).api_key,'replacement-model-key');
    assert.match(element('cfg-llm-status').textContent,/API Key 无效/);
    failSave = true;
    await run('saveSettings()');
    assert.match(element('settings-save-status').textContent,/保存失败/);
    assert.doesNotMatch(element('settings-save-status').textContent,/已保存/);
    assert.equal(element('cfg-llm-key').value,'replacement-model-key');
    failSave = false;
    await run('saveSettings()');
    assert.equal(settingsWrites.at(-1).api_key,'replacement-model-key');
    assert.equal(element('cfg-llm-key').value,'');
    run('closeSettings()');
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
    const oldFetch=context.fetch;
    const regenCalls=[];
    context.fetch=async(url,options)=>{
        regenCalls.push([url,options.method]);
        return {ok:true,json:async()=>({revision:8,can_retract:false})};
    };
    run("currentSessionId='R';chatInner.replaceChildren();appendMessage('user','旧问题','R');let oldReply=appendMessage('assistant','旧回复','R').wrap;appendMessage('user','新问题','R');let lastReply=appendMessage('assistant','新回复','R').wrap;retractions.set('R',{revision:7,can_retract:true});sendMessage=async()=>{window.regenQuery=userInput.value};");
    await run('regenerate(oldReply)');
    assert.equal(regenCalls.length,0);
    await run('regenerate(lastReply)');
    assert.deepEqual(regenCalls,[['/api/conversations/R/messages/last?revision=7','DELETE']]);
    assert.equal(context.window.regenQuery,'新问题');
    assert.equal(element('chat-inner').childNodes.length,2);
    context.fetch=oldFetch;

    let finishOldRequest;
    context.fetch=(url,options)=>url==='/api/conversations/stale/workflow'
        ? new Promise(resolve=>{finishOldRequest=resolve;}) : oldFetch(url,options);
    const stale = run("TaxAuth.request('/api/conversations/stale/workflow')");
    const staleResult = assert.rejects(stale,{name:'AbortError'});
    await run('showSettings()');
    assert.equal(element('cfg-llm-model').value,'personal-model');
    run("switchRole('admin')");
    element('auth-token').value='login-admin-fixture';
    await run('applyAccessToken()');
    await tick();
    finishOldRequest(response({detail:'旧身份已过期'},401));
    await staleResult;
    assert.equal(element('role-label').textContent,'管理员');
    assert.equal(element('settings-modal').open,true);
    assert.equal(element('settings-model-config').hidden,false);
    assert.equal(element('settings-scope-switch').hidden,false);
    assert.equal(element('cfg-llm-model').value,'');
    assert.equal(element('cfg-llm-key').value,'');
    await run("showSettings('system')");
    assert.match(element('cfg-llm-key-state').textContent,/已配置/);
    element('cfg-llm-provider').value='Gemini';
    run("selectProvider('llm')");
    assert.equal(element('cfg-llm-url').value,providers.Gemini);
    element('cfg-llm-model').value='gemini-custom';
    element('cfg-llm-key').value='replacement-model-key';
    await run('saveSettings()');
    assert.equal(settingsWrites.at(-1).api_key,'replacement-model-key');
    assert.match(element('settings-save-status').textContent,/已保存/);
    element('cfg-llm-model').value='gemini-second';
    await run('saveSettings()');
    assert.equal(settingsWrites.at(-1).api_key,undefined);

    // Same-tab refresh uses the same sessionStorage and re-runs the actual page scripts.
    elements.clear(); listeners.clear(); calls.length=0;
    const reloaded = vm.createContext({...context,fetch:oldFetch});
    vm.runInContext(authSource,reloaded);
    vm.runInContext(source,reloaded);
    await tick();
    assert.deepEqual(calls[0],['/api/me','Bearer admin-fixture']);
    assert.equal(element('role-label').textContent,'管理员');
    await vm.runInContext("showSettings('system')",reloaded);
    await tick();
    assert.equal(element('cfg-llm-model').value,'gemini-second');
    assert.equal(element('cfg-llm-key').value,'');
    assert.match(element('cfg-llm-key-state').textContent,/已配置/);
    reloaded.fetch=async()=>response({detail:'访问 Token 无效或已失效'},401);
    await vm.runInContext('loadConversations()',reloaded);
    assert.equal(element('auth-modal').open,true);
    assert.equal(element('settings-modal').open,false);
    assert.match(element('auth-status').textContent,/无效或已失效/);
    assert.equal(reloaded.sessionStorage.getItem('tax_agent_admin_session'),null);
    assert.equal(reloaded.sessionStorage.getItem('tax_agent_user_session'),'user-fixture');
    reloaded.fetch=oldFetch;
    await vm.runInContext("verifyIdentity(() => TaxAuth.useStored('user'))",reloaded);
    await vm.runInContext('showSettings()',reloaded);
    assert.equal(element('cfg-llm-model').value,'personal-model');
    console.log('PASS: 会话流、身份验证与隔离、刷新恢复、失效处理、模型配置保存与重读');
})().catch(error=>{console.error(error);process.exitCode=1;});
