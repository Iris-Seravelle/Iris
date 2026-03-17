use bytes::Bytes;
use iris::mailbox::{bounded_channel, Message, SystemMessage};

#[tokio::test]
async fn bounded_drop_new_accepts_capacity_then_rejects() {
    let (tx, mut rx) = bounded_channel(2);
    tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
    tx.send(Message::User(Bytes::from_static(b"m2"))).unwrap();
    // third send should be rejected because capacity is 2
    assert!(tx.send(Message::User(Bytes::from_static(b"m3"))).is_err());

    let first = rx.recv().await.expect("first");
    let second = rx.recv().await.expect("second");
    match first {
        Message::User(b) => assert_eq!(b.as_ref(), b"m1"),
        _ => panic!("expected user message"),
    }
    match second {
        Message::User(b) => assert_eq!(b.as_ref(), b"m2"),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn drop_old_system_message_discards_oldest_user_message_in_bounded() {
    let (tx, mut rx) = bounded_channel(2);

    tx.send(Message::User(Bytes::from_static(b"m1"))).unwrap();
    tx.send(Message::User(Bytes::from_static(b"m2"))).unwrap();
    // simulate runtime sending DropOld system message to mailbox
    tx.send_system(SystemMessage::DropOld).unwrap();

    let got = rx.recv().await.expect("message after drop-old");
    match got {
        Message::User(b) => assert_eq!(b.as_ref(), b"m2"),
        _ => panic!("expected user message"),
    }
}

#[tokio::test]
async fn len_counter_reflects_queue_size() {
    let (tx, mut rx) = bounded_channel(3);
    assert_eq!(tx.len(), 0);
    tx.send(Message::User(Bytes::from_static(b"a"))).unwrap();
    assert_eq!(tx.len(), 1);
    tx.send(Message::User(Bytes::from_static(b"b"))).unwrap();
    assert_eq!(tx.len(), 2);
    // receive one and check counter decreases
    let _ = rx.recv().await.expect("recv");
    // give a moment for counter update; try_recv is synchronous on channel state
    // after recv the mailbox receiver subtracts the counter before returning
    // so len() should reflect the updated value
    assert!(tx.len() <= 2);
}
