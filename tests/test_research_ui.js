// Logic-level DOM substitutes only: this is not browser/accessibility verification.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const elements = new Map();
class Element {
    constructor(tag='div') {this.tagName=tag; this.children=[]; this.value=''; this.hidden=false; this.disabled=false; this.attributes={}; this._text='';}
    set id(value) {this._id=value; elements.set(value,this);} get id() {return this._id;}
    set textContent(value) {this._text=String(value ?? ''); this.children=[];} get textContent() {return this._text+this.children.map(c=>c.textContent || '').join('');}
    append(...children) {this.children.push(...children);} replaceChildren(...children) {this.children=children; this._text='';}
    add(option) {this.children.push(option);} setAttribute(k,v) {this.attributes[k]=v;} removeAttribute(k) {delete this.attributes[k];}
    focus() {this.focused=true;} querySelectorAll(tag) {return this.children.flatMap(c=>[...(c.tagName===tag ? [c]:[]),...(c.querySelectorAll?.(tag) || [])]);}
}
const html=fs.readFileSync('web/templates/index.html','utf8');
for(const match of html.matchAll(/id="([^"]+)"/g)) {const element=new Element(); element.id=match[1];}
const get=id=>elements.get(id), calls=[], listeners={};
let current={task_id:'test-task',query:'模拟资料',revision:1,status:'created',stage:'created'};
const report={original_query:'模拟资料',notice:'替身测试',confirmed_facts:['模拟资料'],unconfirmed_facts:[],scope:'模拟范围',analysis:[],uncertainties:[],follow_up_questions:[],answer_draft:'不是税务答案。'};
let caseRecord;
function response(value,stream=false) {
    const bytes=new TextEncoder().encode('data: '+JSON.stringify({type:'done',task:value})+'\n\n');
    const parts=[bytes.slice(0,13),bytes.slice(13,24),bytes.slice(24)]; let index=0;
    return {ok:true,json:async()=>value,body:stream ? {getReader:()=>({read:async()=>index<parts.length ? {done:false,value:parts[index++]}:{done:true}})}:undefined};
}
const context={document:{getElementById:get,createElement:tag=>new Element(tag),createTextNode:text=>({textContent:text})},
    Option:function(text,value) {const element=new Element('option');element.textContent=text;element.value=value;return element;},
    window:{addEventListener:(name,fn)=>{listeners[name]=fn;}},AbortController,TextDecoder,
    fetch:async(url,options)=>{
        const body=options.body ? JSON.parse(options.body):null;calls.push({url,body});
        if(url==='/api/research/tasks' && !body) return response([]);
        if(url==='/api/research/tasks') return response(current);
        if(url.endsWith('/stream')) {current={...current,status:'awaiting_review',revision:5,report_version:1,evidence:[],report};return response(current,true);}
        if(url.endsWith('/review')) {assert.equal(body.report_version,1);current={...current,status:'finalized',revision:6};return response(current);}
        if(url.includes('/case-preview')) return response({content:{title:'模拟案例',facts:['模拟资料'],conclusion:'待核实',evidence:[]},source_task_id:current.task_id,source_report_version:1,notice:'预览未保存'});
        if(url==='/api/research/cases' && !body) return response(caseRecord ? [caseRecord]:[]);
        if(url==='/api/research/cases') {caseRecord={case_id:'case',revision:1,current_version:1,status:'draft',content:body.content,save_confirmed:false,professional_review:{status:'not_reviewed'}};return response(caseRecord);}
        if(url==='/api/research/cases/case') {caseRecord={...caseRecord,revision:caseRecord.revision+1,save_confirmed:true,status:body.action==='publish' ? 'published':'draft'};return response(caseRecord);}
        throw Error('Unexpected endpoint '+url);
    }};
const auth = {};
context.TaxAuth = {request:(...args)=>context.fetch(...args),snapshot:()=>auth,user:()=>null};
vm.createContext(context); vm.runInContext(fs.readFileSync('web/static/research.js','utf8'),context);
const settle=async()=>{for(let i=0;i<30;i++) await new Promise(resolve=>setImmediate(resolve));};
const findButton=text=>get('research-result').querySelectorAll('button').find(b=>b.textContent===text);
(async()=>{
    context.window.taxResearchActions.mode(true);await settle();
    assert.equal(get('chat').hidden,true);assert.equal(get('composer').hidden,true);
    get('research-form').onsubmit({preventDefault(){}});await settle();
    assert.equal(get('research-query').attributes['aria-invalid'],'true');
    get('research-query').value='模拟资料';get('research-form').onsubmit({preventDefault(){}});await settle();
    assert.match(get('research-result').textContent,/研究报告/);
    assert.equal(calls.some(c=>c.url.startsWith('/api/chat')),false);
    findButton('确认此版本定稿').onclick();await settle();
    findButton('提取候选案例并预览').onclick();await settle();
    assert.equal(calls.filter(c=>c.url==='/api/research/cases' && c.body).length,0);
    findButton('确认保存并发布到个人案例库').onclick();await settle();
    assert.equal(caseRecord.status,'published');assert.equal(caseRecord.professional_review.status,'not_reviewed');
    assert.deepEqual(calls.filter(c=>c.url==='/api/research/cases/case').map(c=>c.body.action),['consent','publish']);
    listeners['tax-auth-changed']();assert.equal(get('research-result').children.length,0);
    assert.equal(get('research-query').value,'');
    const normalFetch=context.fetch;
    let refreshed=false;
    context.fetch=async(url,options)=>{
        if(url.endsWith('/stream')) {
            const bytes=new TextEncoder().encode('data: '+JSON.stringify({type:'status',status:'running',stage:'assess',elapsed_ms:42000})+'\n\n: keepalive\n\ndata: '+JSON.stringify({type:'error',message:'模拟SSE失败'})+'\n\n');
            let delivered=false;
            return {ok:true,body:{getReader:()=>({read:async()=>delivered ? {done:true}:(delivered=true,{done:false,value:bytes})})}};
        }
        if(url==='/api/research/tasks/test-task') {refreshed=true; return response({...current,status:'failed',stage:'assess',elapsed_ms:42000,resume_stage:'assess'});}
        return normalFetch(url,options);
    };
    get('research-query').value='SSE故障';get('research-form').onsubmit({preventDefault(){}});await settle();
    assert(refreshed,'SSE error must refresh persisted task');
    assert.match(get('research-error').textContent,/模拟SSE失败/);
    assert.match(get('research-status').textContent,/42 秒/);
    assert.equal(get('research-run').textContent,'从已保存步骤恢复');
    context.fetch=normalFetch;
    console.log('PASS: Research mode isolation, validation, fragmented SSE, version confirmation, preview-before-save, consent/publication and identity reset (DOM substitutes)');
})().catch(error=>{console.error(error);process.exitCode=1;});
