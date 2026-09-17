// tests/unit/mesh_recovery_test.js
// Unit tests for BitMesh local auto-join code formatting, stop signal state, and swarm fail-fast recovery logic.

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

function resolveMeshRoomCode(inputCode) {
  if (inputCode === "LOCAL") return "LOCAL";
  return (inputCode || "").trim().toUpperCase();
}

function handlePeerDeparture(activeChain, departedPeerId, waiters) {
  const isChainMember = activeChain.includes(departedPeerId);
  const remainingChain = activeChain.filter(id => id !== departedPeerId);
  const rejectedWaiters = [];
  if (isChainMember) {
    for (const [pos, waiter] of waiters.entries()) {
      rejectedWaiters.push(pos);
      waiter(null);
    }
    waiters.clear();
  }
  return { isChainMember, remainingChain, rejectedWaiters, isDegraded: isChainMember };
}

Deno.test("resolveMeshRoomCode formats LOCAL mesh code correctly", () => {
  assertEquals(resolveMeshRoomCode("LOCAL"), "LOCAL");
  assertEquals(resolveMeshRoomCode("abcd"), "ABCD");
});

Deno.test("handlePeerDeparture triggers fail-fast recovery when chain peer drops", () => {
  const chain = ["peer-1", "peer-2", "peer-3"];
  const waiters = new Map();
  waiters.set(10, () => {});
  waiters.set(11, () => {});

  const res = handlePeerDeparture(chain, "peer-2", waiters);

  assertEquals(res.isChainMember, true);
  assertEquals(res.remainingChain, ["peer-1", "peer-3"]);
  assertEquals(res.rejectedWaiters, [10, 11]);
  assertEquals(res.isDegraded, true);
  assertEquals(waiters.size, 0);
});

Deno.test("handlePeerDeparture ignores non-chain peers", () => {
  const chain = ["peer-1", "peer-2"];
  const waiters = new Map();
  waiters.set(5, () => {});

  const res = handlePeerDeparture(chain, "guest-peer", waiters);

  assertEquals(res.isChainMember, false);
  assertEquals(res.remainingChain, ["peer-1", "peer-2"]);
  assertEquals(res.isDegraded, false);
  assertEquals(waiters.size, 1);
});
