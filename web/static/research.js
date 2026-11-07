/* Research state never shares a chat thread or imports conversation memory. */
(() => {
    'use strict';
    const el = id => document.getElementById(id);
    let task = null, controller = null, busy = false, timer = null, statusAt = Date.now();
    const stages = {created:'待开始', understand:'提取业务事实', retrieve:'检索政策', report:'整理研究报告',
        cases:'比较历史案例', plan:'制定研究计划', assess:'核查研究证据', calculate:'核对计算条件', supplement:'补充研究', prepare_clarification:'等待补充事实',
        finalized:'已确认定稿', awaiting_review:'等待报告复核', awaiting_facts:'等待补充事实', failed:'执行失败', interrupted:'执行已停止，可恢复'};
    const node = (tag, text, parent) => { const n = document.createElement(tag); n.textContent = text; if(parent) parent.append(n); return n; };
    let fieldId=0;
    function field(parent,title,value='',multiline=true) {
        const label=node('label',title,parent), input=node(multiline ? 'textarea':'input','',parent);
        input.id='research-field-'+(++fieldId); label.htmlFor=input.id;
        input.value=Array.isArray(value) ? value.join('\n') : value || '';
        return input;
    }
    const lines=input => input.value.split('\n').map(x => x.trim()).filter(Boolean);
    function caseEditor(parent,content={}) {
        const fields={};
        for(const [key,label,multi] of [['title','研究主题',false],['question','原始问题',true],['facts','业务事实（每行一项）',true],
            ['tax_types','税种（每行一项）',true],['region','地区',false],['tax_period','所属期',false],['business_type','业务类型',false],['enterprise_type','企业类型/纳税人身份',false],
            ['conditions','待重新核验的适用条件（每行一项）',true],['conclusion','历史研究结论',true],['unresolved','未解决事项（每行一项）',true]]) {
            fields[key]=field(parent,label,content[key],multi);
        }
        const sources=node('details','',parent); node('summary',`保留的政策证据（${(content.evidence || []).length}项）`,sources);
        const sourceChecks=(content.evidence || []).map(e => {
            const label=node('label','',sources), check=node('input','',label); check.type='checkbox'; check.checked=true;
            label.append(document.createTextNode(e.title || e.evidence_id)); node('pre',e.content,sources); return {e,check};
        });
        const addition=node('details','',parent); node('summary','补充一项政策资料',addition);
        const sourceTitle=field(addition,'政策名称','',false), sourceUrl=field(addition,'来源网址','',false), sourceText=field(addition,'实际读取的政策节选','',true);
        return () => {
            const value={};
            for(const key of ['title','question','region','tax_period','business_type','enterprise_type','conclusion']) value[key]=fields[key].value;
            for(const key of ['facts','tax_types','conditions','unresolved']) value[key]=lines(fields[key]);
            value.evidence=sourceChecks.filter(x => x.check.checked).map(x => x.e);
            if(sourceText.value.trim()) value.evidence.push({title:sourceTitle.value,source_url:sourceUrl.value,content:sourceText.value});
            return value;
        };
    }
    async function api(path, body, signal) {
        const response = await fetch('/api/research' + path, {method:body ? 'POST':'GET',
            headers:{Authorization:AUTH_TOKEN, 'Content-Type':'application/json'}, body:body ? JSON.stringify(body):undefined, signal});
        if(!response.ok) { const data=await response.json().catch(() => ({})); throw new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请刷新后重试'); }
        return response;
    }
    function setBusy(value) {
        busy=value; window.taxResearchBusy=value;
        for(const id of ['research-create','research-run','research-task-list','research-refresh']) el(id).disabled=value;
        el('research-stop').hidden=!value;
        for(const button of el('research-panel').querySelectorAll('button')) if(button.id !== 'research-stop') button.disabled=value;
    }
    async function list() {
        const rows=await (await api('/tasks')).json(), select=el('research-task-list');
        select.replaceChildren(new Option('新建研究任务',''));
        for(const row of rows) select.add(new Option(`${row.query.slice(0,45)} · ${stages[row.status] || row.status}`,row.task_id));
        select.value=task?.task_id || '';
    }
    function renderStatus() {
        const label=task.status === 'running' ? stages[task.stage] || '正在运行' : stages[task.status] || stages[task.stage] || task.status;
        const activity=task.activity ? ' · ' + task.activity : '';
        el('research-status').textContent=label + activity + ' · 已用 ' + Math.floor(((task.elapsed_ms || 0) + (busy ? Date.now()-statusAt : 0))/1000) + ' 秒 · 任务 ' + task.task_id.slice(0,8);
    }
    function render() {
        const result=el('research-result'); result.replaceChildren();
        el('research-run').hidden=!task || ['finalized','awaiting_review','awaiting_facts'].includes(task.status);
        if(!task) {el('research-status').textContent='创建研究任务后，进度和报告会保存在当前用户下。'; return;}
        renderStatus();
        el('research-run').textContent=task.resume_stage || task.status === 'interrupted' ? '从已保存步骤恢复' : task.status === 'failed' ? '重新研究（从头执行）' : '开始研究';
        const questionCount=task.question_count ?? task.questions?.length;
        if(questionCount) node('p',`初步证据已覆盖 ${task.completed_questions || 0}/${questionCount} 个子问题（仍需人工核验）`,result).id='research-coverage-count';
        if(task.failure) {
            node('h3','尚未生成可复核报告',result);
            node('p',task.failure.message,result);
            const refs=task.failure.details?.unmatched_citations || task.failure.details?.validation?.unmatched_citations;
            if(refs?.length) node('p','未匹配的引用：'+refs.join('、'),result);
            node('p',task.resume_stage ? '已保留证据，可从失败步骤恢复。' : '请查看证据缺口；重新研究将从头执行并重新检索。',result);
        }
        if(task.status === 'awaiting_facts') {
            node('h3','待确认事实',result);
            node('p',task.unknowns.join('\n'),result);
            const label=node('label','补充信息',result), reply=node('textarea','',result);
            reply.id='research-fact-reply'; label.htmlFor=reply.id;
            const send=node('button','补充并继续',result), skip=node('button','保留未知并继续',result);
            const clarify=skipUnknowns => action(() => run({action:'clarify',reply:reply.value,skip_unknowns:skipUnknowns}));
            send.onclick=() => clarify(false); skip.onclick=() => clarify(true);
        }
        if(task.candidates?.length || task.assessment?.coverage?.length) {
            const details=node('details','',result);
            node('summary','检索与证据覆盖详情',details);
            node('p',`读取候选摘录 ${task.candidates?.length || 0} 条，其中 ${task.evidence?.length || 0} 条与子问题关联。`,details);
            for(const item of task.assessment?.coverage || []) {
                node('p',item.question + ' · ' + (item.sufficient ? '已有初步依据' : item.gap || '仍需补检'),details);
            }
        }
        if(task.trace?.some(item => Number.isFinite(item.elapsed_ms))) {
            const timings=node('details','',result); node('summary','已完成步骤与耗时',timings);
            for(const item of task.trace.filter(item => Number.isFinite(item.elapsed_ms))) {
                node('p',`${stages[item.stage] || item.stage} · ${(item.elapsed_ms/1000).toFixed(1)} 秒`,timings);
            }
        }
        if(!task.report) return;
        const report=task.report;
        node('h3',`研究报告 · 第 ${task.report_version} 版`,result); node('p',report.notice,result);
        const block=(title,value) => {node('h3',title,result); node('p',Array.isArray(value) ? value.map(x => '• ' + x).join('\n') || '未提供' : value || '未提供',result);};
        block('原始咨询问题',report.original_query);
        block('业务事实（用户输入，未独立核实）',report.confirmed_facts);
        block('未确认事实',report.unconfirmed_facts); block('研究范围',report.scope);
        block('咨询答复初稿（AI）',report.answer_draft);
        node('h3','核心政策依据',result);
        const sources=report.policy_evidence || task.evidence || [];
        const groups=new Map();
        for(const evidence of sources) {
            const key=evidence.file_id || evidence.evidence_id;
            if(!groups.has(key)) groups.set(key,[]);
            groups.get(key).push(evidence);
        }
        for(const passages of groups.values()) {
            const [evidence]=passages;
            const detail=node('details','',result); node('summary',evidence.title,detail);
            const date=evidence.response_date ? '官方答复日期：'+evidence.response_date :
                evidence.document_date ? '成文日期：'+evidence.document_date : '发布日期：'+(evidence.publish_date || '未知');
            node('p',`发布机关：${evidence.issuer || '未知'}\n${date}\n实施日期：${evidence.effective_date || '未知'}\n地区：${evidence.region || '未知'}\n效力：${evidence.is_current === false || evidence.superseded_by ? '已标记失效或被替代' : evidence.is_current === true ? '来源标记有效，仍需核验' : '未知，待核实'}；业务适用性：待核实`,detail);
            if(/^https?:\/\//i.test(evidence.source_url || '')) {const a=node('a','查看来源',detail); a.href=evidence.source_url; a.target='_blank'; a.rel='noopener noreferrer';}
            for(const passage of passages) node('pre',passage.content,detail);
            const debug=node('details','',detail); node('summary','证据调试详情',debug);
            for(const passage of passages) node('p',passage.evidence_id+' · '+(passage.read_status === 'read_excerpt' ? '已读取摘录':'尚未读取'),debug);
        }
        node('h3','初步适用性分析（AI）',result);
        for(const item of report.analysis) {
            const titles=[...new Set(sources.filter(e => item.evidence_ids.includes(e.evidence_id)).map(e => e.title))];
            node('p',`${item.question}\n${item.finding}\n依据：${titles.join('、') || '缺少依据'}\n${item.qualification || ''}`,result);
        }
        block('不确定事项',report.uncertainties); block('待进一步核实的问题',report.follow_up_questions);
        if(report.case_references?.length) {
            node('h3','历史案例线索与差异',result);
            for(const reference of report.case_references) {
                const detail=node('details','',result); node('summary',`${reference.content.title} · 版本 ${reference.version}`,detail);
                node('p',reference.notice,detail);
                node('p',`专业复核：${reference.professional_review.status === 'reviewed' ? '用户记录已复核（资质未经平台核验）':'尚未专业复核'}`,detail);
                for(const difference of reference.comparisons) node('p',`${{tax_types:'税种',region:'地区',tax_period:'所属期',business_type:'业务类型',enterprise_type:'企业类型'}[difference.field]}：历史 ${difference.historical || '未知'}；当前 ${difference.current || '未知'}；${{same:'相同',different:'存在差异',unknown:'尚无法比较'}[difference.comparison]}`,detail);
                if(reference.policy_comparison) node('p',`共同政策线索：${reference.policy_comparison.shared.join('；') || '暂无'}\n${reference.policy_comparison.notice}`,detail);
                node('p','仍需核对的历史条件：'+(reference.conditions_to_recheck.join('；') || '未知'),detail);
            }
        }
        if(task.status === 'awaiting_review') {
            const button=node('button','确认此版本定稿',result); button.type='button';
            button.onclick=() => action(async () => {task=await (await api(`/tasks/${task.task_id}/review`,
                {revision:task.revision,report_version:task.report_version,action:'confirm'})).json(); render(); await list();});
            node('p','确认不等同于专业审核，也不会自动保存为案例。',result);
            const editing=node('details','',result); node('summary','修改报告或退回研究',editing);
            const scope=field(editing,'研究范围',report.scope), draft=field(editing,'咨询答复初稿',report.answer_draft);
            const uncertainty=field(editing,'不确定事项（每行一项）',report.uncertainties), followUp=field(editing,'待进一步核实的问题（每行一项）',report.follow_up_questions);
            const findings=report.analysis.map(item => ({item,input:field(editing,item.question+'（留空可删除此项分析）',item.finding)}));
            const save=node('button','保存为新版本，继续复核',editing);
            save.onclick=() => action(async () => {task=await (await api(`/tasks/${task.task_id}/review`,
                {revision:task.revision,report_version:task.report_version,action:'revise',report:{scope:scope.value,answer_draft:draft.value,
                    uncertainties:lines(uncertainty),follow_up_questions:lines(followUp),analysis:findings.filter(f => f.input.value.trim()).map(f => ({...f.item,finding:f.input.value}))}})).json(); render(); await list();});
            const reasonLabel=node('label','退回研究意见',editing), reason=node('textarea','',editing);
            reason.id='research-review-reason'; reasonLabel.htmlFor=reason.id;
            const again=node('button','按意见重新研究',editing);
            again.onclick=() => action(() => run({report_version:task.report_version,action:'research_again',reply:reason.value}));
        }
        if(task.status === 'finalized') {
            const button=node('button','提取候选案例并预览',result), target=node('div','',result);
            button.onclick=() => action(async () => {
                const preview=await (await api(`/tasks/${task.task_id}/case-preview?generalize=true`)).json(); target.replaceChildren();
                node('h3','候选案例 · 尚未保存',target); node('p',preview.notice,target);
                const collect=caseEditor(target,preview.content), save=node('button','确认保存并发布到个人案例库',target);
                save.onclick=() => action(async () => {
                    let saved=await (await api('/cases',{...preview,content:collect()})).json();
                    if(!saved.save_confirmed) saved=await (await api(`/cases/${saved.case_id}`,{revision:saved.revision,action:'consent'})).json();
                    if(saved.status !== 'published') await api(`/cases/${saved.case_id}`,{revision:saved.revision,action:'publish'});
                    target.replaceChildren(); node('p','已确认保存并发布。专业复核仍单独记录。',target); await loadCases();
                });
                const cancel=node('button','取消保存',target); cancel.onclick=() => target.replaceChildren();
            });
        }
    }
    async function action(fn) {
        if(busy) return;
        const auth=AUTH_TOKEN;
        setBusy(true); el('research-error').textContent=''; statusAt=Date.now();
        if(typeof setInterval === 'function') timer=setInterval(() => {if(task) renderStatus();},1000);
        try {await fn();}
        catch(error) {
            el('research-error').textContent=error.name === 'AbortError' ? '执行已停止，正在读取保存状态。' : error.message;
            if(task && auth===AUTH_TOKEN) {
                try { task=await (await api(`/tasks/${task.task_id}`)).json(); render(); await list(); }
                catch { el('research-error').textContent+=' 无法读取保存状态，请稍后刷新。'; }
            }
        }
        finally {if(timer !== null) clearInterval(timer); timer=null; controller=null; setBusy(false);}
    }
    async function run(operation={}) {
        controller=new AbortController();
        const id=task.task_id;
        const response=await api(`/tasks/${id}/stream`,{revision:task.revision,...operation},controller.signal);
        const reader=response.body.getReader(), decoder=new TextDecoder(); let buffer='', terminal=false;
        while(true) {
            const {done,value}=await reader.read(); if(done) break;
            buffer+=decoder.decode(value,{stream:true}); let end;
            while((end=buffer.indexOf('\n\n')) >= 0) {
                const frame=buffer.slice(0,end); buffer=buffer.slice(end+2);
                if(frame.startsWith(':')) { if(task) renderStatus(); continue; }
                const line=frame.split('\n').find(x => x.startsWith('data: ')); if(!line) continue;
                const event=JSON.parse(line.slice(6));
                if(event.type==='status') {
                    task={...task,activity:'',...event}; statusAt=Date.now();
                    renderStatus();
                    if(event.question_count) {
                        const count=el('research-coverage-count') || node('p','',el('research-result'));
                        count.id='research-coverage-count';
                        count.textContent=`初步证据已覆盖 ${event.completed_questions}/${event.question_count} 个子问题（仍需人工核验）`;
                    }
                }
                if(event.type==='error') throw new Error(event.message);
                if(event.type==='done') {terminal=true; task=event.task; render(); await list();}
            }
        }
        if(!terminal) {
            task=await (await api(`/tasks/${id}`)).json(); render();
            throw new Error('研究连接已中断，已读取保存的任务状态。');
        }
    }
    async function loadCases() {
        const rows=await (await api('/cases')).json(), container=el('research-cases'); container.replaceChildren();
        if(!rows.length) node('p','还没有个人研究案例。可从定稿报告提取，或人工录入。',container);
        for(const item of rows) {
            const detail=node('details','',container); node('summary',`${item.content.title} · 第${item.current_version}版 · ${{draft:'草稿',published:'已发布',withdrawn:'已撤回'}[item.status]}`,detail);
            node('p',`保存确认：${item.save_confirmed ? '已确认':'尚未确认'}；专业复核：${item.professional_review.status === 'reviewed' ? '用户记录已复核':'未复核'}`,detail);
            const collect=caseEditor(detail,item.content);
            const change=(title,operation,extra=() => ({})) => {
                const button=node('button',title,detail);
                button.onclick=() => action(async () => {await api(`/cases/${item.case_id}`,{revision:item.revision,action:operation,...extra()}); await loadCases();});
            };
            change('保存修改为新版本（需重新确认发布）','edit',() => ({content:collect()}));
            if(!item.save_confirmed) change('确认保存当前版本','consent');
            if(item.status !== 'published' && item.save_confirmed) change('发布当前版本','publish');
            if(item.status === 'published') change('撤回案例','withdraw');
            const reviewNotes=field(detail,'专业复核依据（仅记录用户声明，不验证专业资质）');
            change('记录专业复核','professional_review',() => ({notes:reviewNotes.value}));
            const deletion=node('details','',detail); node('summary','删除案例',deletion);
            node('p','删除将清除案例各版本内容，且不能通过重复保存原报告恢复。',deletion);
            const remove=node('button','确认删除此案例及其版本内容',deletion);
            remove.onclick=() => action(async () => {await api(`/cases/${item.case_id}`,{revision:item.revision,action:'delete'}); await loadCases();});
            const history=node('button','查看版本记录',detail), historyView=node('div','',detail);
            history.onclick=() => action(async () => {const data=await (await api(`/cases/${item.case_id}/history`)).json(); historyView.replaceChildren();
                for(const version of data.versions) node('p',`第${version.version}版：${version.content.title}\n${version.content.conclusion}`,historyView);});
        }
    }
    function resetManual() {
        el('research-case-manual').replaceChildren();
        const manual=caseEditor(el('research-case-manual')), manualSave=node('button','保存人工录入草稿',el('research-case-manual'));
        manualSave.onclick=() => action(async () => {await api('/cases',{content:manual()}); await loadCases();});
    }
    resetManual();
    window.addEventListener('tax-auth-changed',() => {controller?.abort(); task=null; el('research-query').value=''; el('research-result').replaceChildren(); el('research-cases').replaceChildren(); resetManual(); render();});
    el('research-form').onsubmit=event => {event.preventDefault(); action(async () => {
        const input=el('research-query');
        if(!input.value.trim() || input.value.length > 16000) {input.setAttribute('aria-invalid','true'); input.focus(); throw new Error('请输入不超过16000字的研究问题。');}
        input.removeAttribute('aria-invalid');
        el('research-status').textContent='正在创建研究任务…';
        task=await (await api('/tasks',{query:el('research-query').value,web_search:el('research-web').checked})).json(); render(); await run();
    });};
    el('research-task-list').onchange=() => action(async () => {const id=el('research-task-list').value; task=id ? await (await api(`/tasks/${id}`)).json() : null; render();});
    function mode(research) {
        el('research-panel').hidden=!research; el('chat').hidden=research; el('composer').hidden=research;
        el('mode-chat').setAttribute('aria-pressed',String(!research)); el('mode-research').setAttribute('aria-pressed',String(research));
        if(research) action(async () => {render(); await list(); await loadCases();});
    }
    window.taxResearchActions={mode,run:() => action(run),stop:() => controller?.abort(),cases:() => action(loadCases),
        refresh:() => action(async () => {if(task) task=await (await api(`/tasks/${task.task_id}`)).json(); render(); await list();})};
    render();
})();
