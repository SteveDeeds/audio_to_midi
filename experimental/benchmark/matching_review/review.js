"use strict";
const stems = ["bass", "drums", "piano", "guitar", "other", "vocals"];
const $ = id => document.getElementById(id);
let report, reviews, token, record, current, visible = [], saving = false;
let drafts = {}, draftStorageKey, lastReceipt = "";
let contextAudio, contextSources = [], contextController, contextVersion = 0, lastContextPlan;
const contextBuffers = new Map();
const players = ["mix", "clean", "a", "b"].map($);
const colors = {bass: "#98ca80", drums: "#dba668", piano: "#80b8e9", guitar: "#cba0e3"};

function error(message) { $("error").textContent = message; }
function setSaveStatus(message, state = "") {
  for (const id of ["saveBanner", "saveState"]) {
    $(id).textContent = message; $(id).dataset.state = state;
  }
  $("saveBanner").parentElement.dataset.state = state;
}
function connection(ok, message) {
  $("connectionStatus").textContent = message;
  $("connectionStatus").dataset.state = ok ? "" : "error";
}
async function apiJson(path, options = {}, timeout = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {...options, signal: controller.signal});
    if (!response.headers.get("Content-Type")?.includes("application/json"))
      throw new Error(`Server returned HTTP ${response.status}; reconnect to refresh this session.`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
  } finally { clearTimeout(timer); }
}
function commentText(noteId) { return reviews.comments?.[noteId]?.text || ""; }
function selectedStems() {
  return [...document.querySelectorAll("#destinations input:checked")].map(input => input.value);
}
function sameStems(a, b) { return JSON.stringify([...a].sort()) === JSON.stringify([...b].sort()); }
function storeDrafts() {
  try { localStorage.setItem(draftStorageKey, JSON.stringify(drafts)); }
  catch (failure) { error(`Browser draft backup failed: ${failure.message}. Keep this page open until you save.`); }
}
function pruneDrafts() {
  for (const [id, draft] of Object.entries(drafts)) {
    const decision = reviews.decisions[id];
    if (Object.hasOwn(draft, "pendingStatus")) {
      const status = draft.pendingStatus;
      if (status === null && !decision ||
          decision?.scope === "exact_note" && decision.status === status &&
          (status !== "confirmed" || sameStems(draft.stems, decision.stems))) {
        delete draft.pendingStatus;
        if (status !== "confirmed") draft.stems = decision?.stems || [];
      }
    }
    if (!Object.hasOwn(draft, "pendingStatus") &&
        sameStems(draft.stems, decision?.stems || []) && draft.comment === commentText(id))
      delete drafts[id];
  }
}
function captureDraft() {
  if (!current) return;
  drafts[current.id] = {...drafts[current.id], stems: selectedStems(), comment: $("noteComment").value};
  pruneDrafts(); storeDrafts(); renderSaveState();
}
function renderSaveState() {
  const count = Object.keys(drafts).length;
  if (current && drafts[current.id]) {
    setSaveStatus(`${lastReceipt ? lastReceipt + " " : ""}UNSAVED changes for ${current.family} ${current.pitch} at ${current.onset.toFixed(3)}s. ` +
      `Click Save exact-note review or Save comment only. ${count} note(s) have browser drafts.`, "unsaved");
  } else if (count) {
    setSaveStatus(`${count} note(s) have UNSAVED browser drafts. Choose the Unsaved drafts queue.`, "unsaved");
  } else {
    setSaveStatus(lastReceipt || "No new save confirmed this session. Use Save exact-note review.", lastReceipt ? "saved" : "");
  }
  if (current && !drafts[current.id]) {
    const decision = reviews.decisions[current.id], comment = reviews.comments?.[current.id];
    if (decision?.scope === "exact_note")
      $("saveState").textContent = `Saved exact-note review: ${decision.status} ${decision.stems.join(", ")}. ` +
        (decision.saved_at ? `Saved ${new Date(decision.saved_at).toLocaleString()}.` : "");
    else $("saveState").textContent = "Exact-note review NOT saved for this note. Choose stems and click Save exact-note review.";
    if (comment) $("saveState").textContent += ` Comment saved ${new Date(comment.saved_at).toLocaleString()}.`;
  }
  if (!current) $("saveState").textContent = "No note selected. Choose another queue or song.";
}
function setBusy(value) {
  saving = value;
  for (const element of document.querySelectorAll(
    "#song,#family,#queue,#previous,#next,#refreshSaved,#noteComment,#destinations input,#confirm,#absent,#uncertain,#clear,#saveComment"))
    element.disabled = value;
  if (!value && !current)
    for (const id of ["noteComment", "confirm", "absent", "uncertain", "clear", "saveComment"]) $(id).disabled = true;
}
async function checkConnection() {
  try {
    const health = await apiJson("/health", {}, 5000);
    connection(true, health.revision !== reviews.revision && !saving
      ? "Server is online, but saved reviews changed elsewhere. Refresh saved data before saving."
      : `Server online. Loaded saved revision ${reviews.revision}.`);
  } catch (failure) { connection(false, `Server connection failed: ${failure.message}. Drafts are NOT saved reviews.`); }
}
async function refreshSavedState() {
  if (saving) return;
  captureDraft(); const noteId = current?.id; setBusy(true);
  try {
    const data = await apiJson("/report.json", {}, 15000);
    if (data.reviews.report_sha256 !== reviews.report_sha256) throw new Error("Pilot changed; do not merge these drafts.");
    reviews = data.reviews; token = data.token;
    pruneDrafts(); storeDrafts(); updateQueue(noteId);
    connection(true, `Server online. Reloaded saved revision ${reviews.revision}.`);
  } catch (failure) { error(`Could not refresh saved data: ${failure.message}`); connection(false, "Saved data could not be refreshed."); }
  finally { setBusy(false); }
}
function cell(row, text) {
  const node = document.createElement("td");
  node.textContent = text;
  row.append(node);
}
function stopContext() {
  contextVersion++;
  if (contextController) contextController.abort();
  for (const source of contextSources) source.stop();
  contextSources = [];
  $("contextStatus").textContent = "Context stopped.";
}
function stop() { stopContext(); players.forEach(player => player.pause()); }
function passageBounds() {
  return [Math.max(0, current.onset - .7),
    Math.min(record.duration, Math.max(current.onset + 1.8, current.offset + .3))];
}
function updateSoundStatus() {
  const silent = players.filter(player => player.muted || player.volume === 0);
  $("playbackStatus").textContent = silent.length
    ? `Player sound is off: ${silent.map(player => player.id).join(", ")}. Use Restore player sound or a replay button.`
    : "Player sound is enabled. Replay buttons restart the highlighted note's passage.";
}
function enableSound(player) {
  if (player.muted) player.muted = false;
  if (player.volume === 0) player.volume = 1;
  updateSoundStatus();
}
async function contextBuffer(path, signal) {
  if (contextBuffers.has(path)) {
    const cached = contextBuffers.get(path);
    contextBuffers.delete(path); contextBuffers.set(path, cached);
    return cached;
  }
  const response = await fetch(path, {signal, headers: {"X-Review-Token": token}});
  if (!response.ok) throw new Error(`Audio request failed (${response.status}): ${await response.text()}`);
  const buffer = await contextAudio.decodeAudioData(await response.arrayBuffer());
  contextBuffers.set(path, buffer);
  while (contextBuffers.size > 4) contextBuffers.delete(contextBuffers.keys().next().value);
  return buffer;
}
function bufferPeak(buffer, start, duration) {
  let peak = 0;
  const first = Math.max(0, Math.floor(start * buffer.sampleRate));
  const last = Math.min(buffer.length, Math.ceil((start + duration) * buffer.sampleRate));
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const samples = buffer.getChannelData(channel);
    for (let i = first; i < last; i++) peak = Math.max(peak, Math.abs(samples[i]));
  }
  return peak;
}
async function playContext(kind) {
  if (!current) return;
  stop();
  const version = contextVersion, note = current, song = record;
  const [start, end] = passageBounds();
  const stem = kind === "cue" ? null : $(kind === "a" ? "stemA" : "stemB").value;
  const controller = new AbortController(); contextController = controller;
  error(""); $("contextStatus").textContent = "Preparing synchronized context audio...";
  try {
    if (!contextAudio) contextAudio = new AudioContext();
    await contextAudio.resume();
    const candidatePath = kind === "cue" ? `/cue/${encodeURIComponent(note.id)}.wav` :
      `/audio/${song.clip}/${stem}.wav`;
    const [target, timing] = await Promise.all([
      contextBuffer(candidatePath, controller.signal),
      apiJson(`/click/${encodeURIComponent(song.clip)}.json`),
    ]);
    if (version !== contextVersion) return;
    const targetOffset = kind === "cue" ? 0 : start;
    const targetDuration = Math.min(target.duration - targetOffset,
      kind === "cue" ? end - note.onset : end - start);
    if (targetDuration <= 0) throw new Error("Target audio does not cover this passage");
    const clickGain = Number($("clickLevel").value) / 100;
    const beats = timing.beats.filter(beat => beat.time >= start && beat.time < end);
    const clickBuffer = accent => {
      const buffer = contextAudio.createBuffer(1, Math.ceil(.04 * contextAudio.sampleRate), contextAudio.sampleRate);
      const samples = buffer.getChannelData(0), frequency = accent ? 1800 : 1200;
      for (let i = 0; i < samples.length; i++) {
        const time = i / buffer.sampleRate;
        samples[i] = .25 * Math.sin(2 * Math.PI * frequency * time) *
          Math.min(1, time / .001) * Math.exp(-time * 160) * (1 - i / samples.length);
      }
      return buffer;
    };
    const tick = clickBuffer(true), toc = clickBuffer(false);
    let firstClick = 0, overlappingClicks = 0;
    for (let i = 0; i < beats.length; i++) {
      while (beats[i].time - beats[firstClick].time >= tick.duration) firstClick++;
      overlappingClicks = Math.max(overlappingClicks, i - firstClick + 1);
    }
    const targetPeak = bufferPeak(target, targetOffset, targetDuration);
    const peakBound = Math.SQRT2 * (.25 * overlappingClicks * clickGain + targetPeak);
    const masterGain = Math.min(.7, .89 / Math.max(peakBound, 1e-8));
    const when = contextAudio.currentTime + .12;
    const plan = {
      noteId: note.id, kind, stem, start, end, when, masterGain, peakBound: peakBound * masterGain,
      sources: [],
    };
    const schedule = (buffer, name, offset, duration, delay, pan, gain) => {
      const source = contextAudio.createBufferSource(), level = contextAudio.createGain();
      const panner = contextAudio.createStereoPanner();
      source.buffer = buffer; level.gain.value = gain * masterGain; panner.pan.value = pan;
      source.connect(level).connect(panner).connect(contextAudio.destination);
      source.onended = () => {
        source.disconnect(); level.disconnect(); panner.disconnect();
        contextSources = contextSources.filter(item => item !== source);
        if (version === contextVersion && !contextSources.length) $("contextStatus").textContent = "Context finished. Replay or compare the next stem.";
      };
      source.start(when + delay, offset, duration);
      contextSources.push(source);
      plan.sources.push({name, when: when + delay, offset, duration, pan, gain: gain * masterGain});
    };
    for (const beat of beats) {
      const buffer = beat.accent ? tick : toc;
      schedule(buffer, beat.accent ? "tick" : "toc", 0, Math.min(buffer.duration, end - beat.time),
        beat.time - start, -.25, clickGain);
    }
    schedule(target, kind === "cue" ? "isolated_note" : stem, targetOffset, targetDuration,
      kind === "cue" ? note.onset - start : 0, .25, 1);
    lastContextPlan = plan;
    $("contextStatus").textContent =
      `Playing MIDI-timed tick/toc clicks slightly left and ${kind === "cue" ? "only the isolated target note" : `stem ${stem}`} slightly right. ` +
        `Target note starts ${(note.onset - start).toFixed(2)}s into this passage.`;
  } catch (failure) {
    if (version !== contextVersion) return;
    stopContext();
    error(`Context playback failed: ${failure.message}`);
  }
}
function isSuspect(note) {
  return note.review_reasons.length || !note.suggested_stems.includes(note.legacy_stem);
}
function reviewSample() {
  const selected = new Set();
  const severity = note => note.review_reasons.length +
    (note.suggested_stems.length && note.suggested_stems[0] !== note.legacy_stem ? 2 : 0);
  const bestScore = note => Math.max(...stems.map(stem => note.evidence.stems[stem].score));
  for (const family of ["bass", "drums", "piano", "guitar"]) {
    const notes = record.notes.filter(note => note.family === family);
    if (!notes.length) continue;
    notes.sort((a, b) => severity(b) - severity(a) || bestScore(a) - bestScore(b));
    const hard = notes[0]; selected.add(hard.id);
    const others = notes.filter(note => note.id !== hard.id);
    others.sort((a, b) => severity(a) - severity(b) || bestScore(b) - bestScore(a));
    const clear = others.find(note => Math.abs(note.onset - hard.onset) >= 2) || others[0];
    if (clear) selected.add(clear.id);
  }
  return selected;
}
function updateSummary() {
  const decisions = Object.values(reviews.decisions);
  const exact = decisions.filter(decision => decision.scope === "exact_note").length;
  $("summary").textContent = `${report.summary.source_groups} original training sources; ` +
    `${report.summary.notes} notes; ${exact} note-level reviewed; ${decisions.length - exact} earlier decisions need note-level recheck. ` +
    `${Object.keys(reviews.comments || {}).length} saved comments. ` +
    "No model has been trained on these suggestions.";
}
function updateQueue(preferred) {
  const family = $("family").value, queue = $("queue").value;
  const sample = queue === "sample" ? reviewSample() : new Set();
  visible = record.notes.filter(note => {
    const decision = reviews.decisions[note.id], reviewed = decision?.scope === "exact_note";
    return (family === "all" || family === note.family) &&
      (queue === "all" || queue === "sample" && sample.has(note.id) || queue === "suspect" && isSuspect(note) ||
       queue === "unreviewed" && !reviewed || queue === "reviewed" && reviewed ||
       queue === "recheck" && decision && !reviewed || queue === "drafts" && Object.hasOwn(drafts, note.id));
  });
  $("queueCount").textContent = `${visible.length} notes in this queue`;
  const body = $("notes").querySelector("tbody");
  body.replaceChildren();
  for (const note of visible) {
    const row = document.createElement("tr");
    row.dataset.noteId = note.id;
    cell(row, note.onset.toFixed(3));
    cell(row, `${note.family} / ${note.pitch}`);
    cell(row, note.legacy_stem);
    cell(row, note.suggested_stems.join(", ") || "none");
    const decision = reviews.decisions[note.id];
    cell(row, decision ? `${decision.scope === "exact_note" ? "" : "RECHECK NOTE: "}${decision.status} ${decision.stems.join(", ")}` : "unreviewed");
    row.addEventListener("click", () => { if (!saving) selectNote(note); });
    body.append(row);
  }
  selectNote(visible.find(note => note.id === preferred) || visible[0]);
  updateSummary();
}
function setAudio(player, name) {
  const path = `/audio/${record.clip}/${name}.wav`;
  if (player.getAttribute("src") !== path) player.src = path;
}
function drawRoll() {
  const canvas = $("roll"), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!current) return;
  const left = Math.max(0, current.onset - 1.5);
  const right = Math.min(record.duration, Math.max(current.onset + 2.5, current.offset + .5));
  const notes = record.notes.filter(note => note.offset >= left && note.onset <= right);
  const low = Math.min(...notes.map(note => note.pitch)) - 2;
  const high = Math.max(...notes.map(note => note.pitch)) + 2;
  for (let pitch = low; pitch <= high; pitch++) {
    const y = 195 - (pitch - low) / (high - low) * 175;
    ctx.strokeStyle = "#203246";
    ctx.beginPath(); ctx.moveTo(35, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    if (pitch % 12 === 0) {
      ctx.fillStyle = "#abb7ca"; ctx.fillText(String(pitch), 0, y);
    }
  }
  for (const note of notes) {
    const x = 35 + (Math.max(left, note.onset) - left) / (right - left) * 1160;
    const width = Math.max(3, (Math.min(right, note.offset) - Math.max(left, note.onset)) / (right - left) * 1160);
    const y = 195 - (note.pitch - low) / (high - low) * 175;
    ctx.fillStyle = note.id === current.id ? "#ff746b" : colors[note.family];
    ctx.fillRect(x, y - 4, width, 7);
  }
  ctx.fillStyle = "#edf2fa";
  ctx.fillText(`${left.toFixed(2)}s`, 35, 215);
  ctx.fillText(`${right.toFixed(2)}s`, 1130, 215);
}
function selectNote(note) {
  stop(); current = note;
  document.querySelectorAll("#notes tbody tr").forEach(row => {
    row.classList.toggle("selected", note && row.dataset.noteId === note.id);
  });
  if (!note) {
    $("noteTitle").textContent = "No notes in this queue";
    $("noteDetails").textContent = "";
    $("reasons").textContent = "";
    $("evidence").querySelector("tbody").replaceChildren();
    document.querySelectorAll("#destinations input").forEach(input => { input.checked = false; });
    $("noteComment").value = "";
    $("saveState").textContent = "No note selected. Choose another queue or song.";
    setBusy(saving); renderSaveState();
    drawRoll(); return;
  }
  $("noteTitle").textContent = `${note.family} MIDI ${note.pitch}, onset ${note.onset.toFixed(3)}s`;
  const level = note.evidence.clean_rms_dbfs;
  $("noteDetails").textContent = `Source velocity ${note.velocity}/127; MIDI duration ` +
    `${((note.offset - note.onset) * 1000).toFixed(0)} ms; clean onset-window level ${level.toFixed(1)} dBFS RMS. ` +
    (level < -60 ? "Very quiet reference: skip or mark uncertain if inaudible." :
      "These are source measurements, not proof that the note survived separation.");
  $("reasons").textContent = note.review_reasons.join("; ") || "No additional heuristic warning; still requires review.";
  const ranked = [...stems].sort((a, b) => note.evidence.stems[b].score - note.evidence.stems[a].score);
  $("stemA").value = ranked[0]; $("stemB").value = ranked[1];
  setAudio($("mix"), "mix"); setAudio($("clean"), `clean_${note.family}`);
  setAudio($("a"), $("stemA").value); setAudio($("b"), $("stemB").value);
  const body = $("evidence").querySelector("tbody"); body.replaceChildren();
  for (const stem of ranked) {
    const row = document.createElement("tr"), evidence = note.evidence.stems[stem];
    cell(row, stem);
    for (const key of ["score", "spectral_similarity", "estimated_retention", "fit_residual"])
      cell(row, evidence[key].toFixed(3));
    cell(row, note.model_support[stem].map(hit =>
      `${hit.model} (${(hit.onset_error_seconds * 1000).toFixed(0)}ms)`).join("; ") || "No matched cached prediction");
    body.append(row);
  }
  const decision = reviews.decisions[note.id];
  document.querySelectorAll("#destinations input").forEach(input => {
    input.checked = (drafts[note.id]?.stems || decision?.stems || []).includes(input.value);
  });
  $("noteComment").value = drafts[note.id]?.comment ?? commentText(note.id);
  setBusy(saving); renderSaveState();
  drawRoll();
}
async function play(player) {
  if (!current) return;
  stop();
  enableSound(player);
  player.currentTime = passageBounds()[0];
  try { await player.play(); } catch (failure) { error(`Playback failed: ${failure.message}`); }
}
async function save(status, commentOnly = false) {
  if (!current || saving) return;
  const selected = selectedStems();
  if (!commentOnly && status === "confirmed" && !selected.length) {
    error("Select at least one audible stem, or explicitly save uncertain/absent.");
    setSaveStatus("NOT SAVED: choose a destination or use Save: uncertain / mask.", "error"); return;
  }
  captureDraft();
  const noteId = current.id, label = `${current.family} ${current.pitch} at ${current.onset.toFixed(3)}s`;
  const decision = status === null ? null : {status, stems: status === "confirmed" ? selected : [], scope: "exact_note"};
  const request = {note_id: noteId, revision: reviews.revision};
  if (!commentOnly) {
    request.decision = decision;
    drafts[noteId] = {stems: selected, comment: $("noteComment").value, pendingStatus: status};
  }
  if (commentOnly || status !== null) request.comment = $("noteComment").value;
  storeDrafts(); setBusy(true); error(""); setSaveStatus(`Saving ${label}...`);
  try {
    const data = await apiJson("/reviews", {
      method: "POST", headers: {"Content-Type": "application/json", "X-Review-Token": token},
      body: JSON.stringify(request),
    });
    reviews = data;
    if (!commentOnly && status !== null) delete drafts[noteId];
    else if (!commentOnly && drafts[noteId]) {
      delete drafts[noteId].pendingStatus;
      drafts[noteId].stems = [];
    }
    pruneDrafts(); storeDrafts();
    lastReceipt = `Saved ${commentOnly ? "comment" : status === null ? "review removal" : "exact-note review"} for ${label} ` +
      `at ${new Date(reviews.updated_at).toLocaleString()} (server revision ${reviews.revision}).`;
    updateQueue(noteId); renderSaveState();
    connection(true, `Server online. Save confirmed on disk: revision ${reviews.revision}.`);
  } catch (failure) {
    error(`Save could not be confirmed: ${failure.message}. Your browser draft is retained; reconnect to check saved data.`);
    setSaveStatus(`SAVE NOT CONFIRMED for ${label}. Draft retained; refresh saved data before retrying.`, "error");
    connection(false, "Save failed or connection was lost. Do not assume this review was saved.");
  } finally { setBusy(false); }
}
async function initialize() {
  if (location.protocol === "file:") throw new Error(
    "Start the matching_pilot serve command and open http://127.0.0.1:8767. Direct file mode cannot save reviews or stream dataset audio.");
  ({report, reviews, token} = await apiJson("/report.json", {}, 15000));
  draftStorageKey = `matching-review-drafts:${reviews.report_sha256}`;
  const stored = localStorage.getItem(draftStorageKey);
  if (stored) {
    drafts = JSON.parse(stored);
    if (!drafts || typeof drafts !== "object" || Array.isArray(drafts)) throw new Error("Invalid browser draft backup; keep it for recovery.");
    for (const draft of Object.values(drafts))
      if (!draft || !Array.isArray(draft.stems) || typeof draft.comment !== "string")
        throw new Error("Invalid browser draft entry; keep it for recovery.");
    pruneDrafts(); storeDrafts();
  }
  for (const item of report.records) {
    const option = document.createElement("option");
    option.value = item.clip; option.textContent = item.composition_id;
    $("song").append(option);
  }
  for (const stem of stems) {
    for (const select of [$("stemA"), $("stemB")]) {
      const option = document.createElement("option"); option.value = stem; option.textContent = stem;
      select.append(option);
    }
    const label = document.createElement("label"), input = document.createElement("input");
    input.type = "checkbox"; input.value = stem; input.setAttribute("aria-label", `Confirm ${stem}`);
    input.addEventListener("change", captureDraft);
    label.append(input, document.createTextNode(stem)); $("destinations").append(label);
  }
  record = report.records[0]; updateQueue();
  $("song").addEventListener("change", () => {
    record = report.records.find(item => item.clip === $("song").value); updateQueue();
  });
  for (const id of ["family", "queue"]) $(id).addEventListener("change", () => updateQueue());
  for (const [id, direction] of [["previous", -1], ["next", 1]]) $(id).addEventListener("click", () => {
    const index = visible.indexOf(current) + direction;
    if (index >= 0 && index < visible.length) selectNote(visible[index]);
  });
  $("stemA").addEventListener("change", () => { stop(); setAudio($("a"), $("stemA").value); });
  $("stemB").addEventListener("change", () => { stop(); setAudio($("b"), $("stemB").value); });
  for (const [button, player] of [["playMix", "mix"], ["playClean", "clean"], ["playA", "a"], ["playB", "b"]])
    $(button).addEventListener("click", () => play($(player)));
  $("stop").addEventListener("click", stop);
  $("noteComment").addEventListener("input", captureDraft);
  $("saveComment").addEventListener("click", () => save(undefined, true));
  $("refreshSaved").addEventListener("click", refreshSavedState);
  window.addEventListener("beforeunload", event => {
    if (saving || Object.keys(drafts).length) { event.preventDefault(); event.returnValue = ""; }
  });
  window.addEventListener("offline", () => connection(false, "Browser is offline. Edits are drafts, not saved reviews."));
  window.addEventListener("online", checkConnection);
  setInterval(checkConnection, 15000);
  checkConnection();
  $("contextStop").addEventListener("click", stop);
  for (const [id, kind] of [["contextCue", "cue"], ["contextA", "a"], ["contextB", "b"]])
    $(id).addEventListener("click", () => playContext(kind));
  $("clickLevel").addEventListener("input", stopContext);
  $("restoreSound").addEventListener("click", () => {
    for (const player of players) { player.muted = false; player.volume = 1; }
    updateSoundStatus();
  });
  for (const status of ["confirmed", "absent", "uncertain"])
    $(status === "confirmed" ? "confirm" : status).addEventListener("click", () => save(status));
  $("clear").addEventListener("click", () => save(null));
  $("download").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(reviews, null, 2)], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = "reviews.json"; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  for (const player of players) {
    player.addEventListener("play", () => {
      stopContext();
      players.filter(other => other !== player).forEach(other => other.pause());
      if (current) {
        const [start, end] = passageBounds();
        if (player.currentTime < start || player.currentTime >= end) player.currentTime = start;
      }
      updateSoundStatus();
    });
    player.addEventListener("timeupdate", () => {
      if (current && player.currentTime >= passageBounds()[1]) player.pause();
    });
    player.addEventListener("volumechange", updateSoundStatus);
    player.addEventListener("error", () => error("Audio could not be loaded; check that the local server and source files are available."));
  }
  updateSoundStatus();
}
initialize().catch(failure => {
  error(failure.message);
  setSaveStatus("Could not load saved reviews. No new save is confirmed.", "error");
  connection(false, "Server unavailable or saved data could not be loaded. Reconnect before reviewing.");
});
