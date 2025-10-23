import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;

public class GenerateRealisticLogs {

    private static final long TOTAL_LOGS = 1_000_000L;
    private static final int CHUNK_SIZE = 100_000;
    private static final String OUTPUT_FILE = "log_classifier/data/logs.csv";
    private static final Random random = new Random();

    // Much more diverse vocabulary
    private static final String[] APPROVAL_VERBS = {
        "approved", "validated", "authorized", "confirmed", "granted", "accepted",
        "signed off", "ratified", "endorsed", "certified", "sanctioned", "cleared",
        "permitted", "allowed", "okayed", "greenlit", "passed", "verified",
        "endorsed", "ratified", "sanctioned", "cleared", "permitted", "allowed",
        "okayed", "greenlit", "passed", "verified", "endorsed", "ratified"
    };

    private static final String[] APPROVAL_NOUNS = {
        "request", "transaction", "payment", "document", "application", "proposal",
        "submission", "order", "invoice", "claim", "ticket", "workflow", "process",
        "change", "deployment", "release", "access", "permission", "update"
    };

    private static final String[] ACK_VERBS = {
        "acknowledged", "received", "confirmed receipt", "got", "captured",
        "registered", "logged", "recorded", "noted", "processed", "accepted",
        "queued", "delivered", "sent back", "replied to", "confirmed", "verified",
        "validated", "checked", "reviewed", "monitored", "tracked", "followed",
        "responded to", "replied", "answered", "reacted to", "handled", "managed"
    };

    private static final String[] ACK_NOUNS = {
        "message", "request", "packet", "signal", "event", "notification",
        "ping", "heartbeat", "response", "reply", "data", "payload", "command",
        "instruction", "query", "call", "webhook", "alert", "complaint", "issue",
        "report", "feedback", "complaint", "inquiry", "ticket", "case", "matter"
    };

    private static final String[] ERROR_VERBS = {
        "failed", "crashed", "errored", "threw exception", "timed out",
        "rejected", "denied", "blocked", "terminated", "aborted", "halted",
        "stopped", "broke", "malfunctioned", "hung", "stalled", "lost",
        "disconnected", "unavailable", "unreachable", "corrupted", "damaged",
        "inaccessible", "unresponsive", "deadlocked", "overflowed", "underflowed"
    };

    private static final String[] ERROR_NOUNS = {
        "connection", "database", "server", "service", "API", "endpoint",
        "process", "thread", "operation", "query", "transaction", "request",
        "handler", "controller", "module", "component", "system", "hardware",
        "printer", "scanner", "network", "connectivity", "device", "equipment",
        "peripheral", "interface", "protocol", "session", "cache", "memory"
    };

    private static final String[] USERS = {
        "alice", "bob", "charlie", "admin", "system", "user123", "john_doe",
        "support_team", "dev_ops", "qa_tester", "manager", "client_456"
    };

    private static final String[] SYSTEMS = {
        "PaymentService", "AuthAPI", "Database", "CacheLayer", "MessageQueue",
        "LoadBalancer", "APIGateway", "FileStorage", "Analytics", "ReportEngine"
    };

    // Generate much more varied patterns with challenging edge cases
    private static String makeApprovalLog() {
        int pattern = random.nextInt(15);
        
        switch (pattern) {
            case 0: return String.format("%s %s %s", 
                rand(USERS), rand(APPROVAL_VERBS), rand(APPROVAL_NOUNS));
            case 1: return String.format("%s %s successfully", 
                rand(APPROVAL_NOUNS), rand(APPROVAL_VERBS));
            case 2: return String.format("%s: %s %s by %s",
                rand(SYSTEMS), rand(APPROVAL_NOUNS), rand(APPROVAL_VERBS), rand(USERS));
            case 3: return String.format("Request #%d %s", 
                random.nextInt(9999), rand(APPROVAL_VERBS));
            case 4: return String.format("%s completed: %s %s",
                rand(APPROVAL_VERBS), rand(APPROVAL_NOUNS), "granted");
            case 5: return String.format("%s granted %s for %s",
                rand(USERS), rand(APPROVAL_NOUNS), rand(USERS));
            case 6: return String.format("Successfully %s %s in %s",
                rand(APPROVAL_VERBS), rand(APPROVAL_NOUNS), rand(SYSTEMS));
            case 7: return String.format("%s: approval status = %s",
                rand(APPROVAL_NOUNS), rand(APPROVAL_VERBS));
            case 8: return String.format("%s has been %s by team",
                rand(APPROVAL_NOUNS), rand(APPROVAL_VERBS));
            case 9: return String.format("Store supervisor %s refund for customer",
                rand(APPROVAL_VERBS));
            case 10: return String.format("Manager %s employee discount request",
                rand(APPROVAL_VERBS));
            case 11: return String.format("System %s credit card payment processing",
                rand(APPROVAL_VERBS));
            case 12: return String.format("Cashier %s split payment transaction",
                rand(APPROVAL_VERBS));
            case 13: return String.format("Multi-step transaction %s by admin",
                rand(APPROVAL_VERBS));
            case 14: return String.format("Admin %s the %s",
                rand(APPROVAL_VERBS), rand(APPROVAL_NOUNS));
            default: return String.format("Admin %s the %s",
                rand(APPROVAL_VERBS), rand(APPROVAL_NOUNS));
        }
    }

    private static String makeAckLog() {
        int pattern = random.nextInt(15);
        
        switch (pattern) {
            case 0: return String.format("%s %s from %s",
                rand(ACK_VERBS), rand(ACK_NOUNS), rand(USERS));
            case 1: return String.format("System %s %s",
                rand(ACK_VERBS), rand(ACK_NOUNS));
            case 2: return String.format("%s: %s %s",
                rand(SYSTEMS), rand(ACK_NOUNS), rand(ACK_VERBS));
            case 3: return String.format("Response %s for request #%d",
                rand(ACK_VERBS), random.nextInt(9999));
            case 4: return String.format("%s %s successfully",
                rand(ACK_NOUNS), rand(ACK_VERBS));
            case 5: return String.format("Device %s %s from server",
                rand(ACK_VERBS), rand(ACK_NOUNS));
            case 6: return String.format("%s after %d retries",
                rand(ACK_VERBS), random.nextInt(5) + 1);
            case 7: return String.format("Client confirmed: %s %s",
                rand(ACK_NOUNS), rand(ACK_VERBS));
            case 8: return String.format("%s acknowledgement sent",
                rand(ACK_NOUNS));
            case 9: return String.format("Customer %s receipt of refund notification",
                rand(ACK_VERBS));
            case 10: return String.format("Staff %s customer complaint about service",
                rand(ACK_VERBS));
            case 11: return String.format("System %s order cancellation request",
                rand(ACK_VERBS));
            case 12: return String.format("Customer service rep %s account information",
                rand(ACK_VERBS));
            case 13: return String.format("Support team %s technical issue report",
                rand(ACK_VERBS));
            case 14: return String.format("Received and %s %s",
                rand(ACK_VERBS), rand(ACK_NOUNS));
            default: return String.format("Received and %s %s",
                rand(ACK_VERBS), rand(ACK_NOUNS));
        }
    }

    private static String makeErrorLog() {
        int pattern = random.nextInt(15);
        
        switch (pattern) {
            case 0: return String.format("Error: %s %s",
                rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 1: return String.format("%s %s during operation",
                rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 2: return String.format("%s: Critical - %s %s",
                rand(SYSTEMS), rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 3: return String.format("Exception in %s: %s",
                rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 4: return String.format("%s encountered while processing",
                rand(ERROR_VERBS));
            case 5: return String.format("Error code %d: %s %s",
                500 + random.nextInt(4), rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 6: return String.format("%s threw error in %s",
                rand(ERROR_NOUNS), rand(SYSTEMS));
            case 7: return String.format("Critical: %s %s - retrying",
                rand(ERROR_NOUNS), rand(ERROR_VERBS));
            case 8: return String.format("System %s after %s failure",
                rand(ERROR_VERBS), rand(ERROR_NOUNS));
            case 9: return String.format("Database connection %s",
                rand(ERROR_VERBS));
            case 10: return String.format("Barcode scanner hardware %s",
                rand(ERROR_VERBS));
            case 11: return String.format("Network connectivity %s",
                rand(ERROR_VERBS));
            case 12: return String.format("Printer %s detected",
                rand(ERROR_VERBS));
            case 13: return String.format("CRITICAL ERROR: Payment gateway %s",
                rand(ERROR_VERBS));
            case 14: return String.format("Network connectivity issue %s after router restart",
                rand(ERROR_VERBS));
            case 15: return String.format("Server %s maintenance window completion",
                rand(ERROR_VERBS));
            case 16: return String.format("System %s backup completion for database",
                rand(ERROR_VERBS));
            case 17: return String.format("%s - %s %s",
                rand(SYSTEMS), rand(ERROR_NOUNS), rand(ERROR_VERBS));
            default: return String.format("%s - %s %s",
                rand(SYSTEMS), rand(ERROR_NOUNS), rand(ERROR_VERBS));
        }
    }

    private static String rand(String[] arr) {
        return arr[random.nextInt(arr.length)];
    }

    private static String addVariation(String msg) {
        // Add realistic variations
        if (random.nextDouble() < 0.3) {
            msg = msg.toLowerCase();
        } else if (random.nextDouble() < 0.1) {
            msg = msg.toUpperCase();
        }
        
        // Add prefixes sometimes
        if (random.nextDouble() < 0.2) {
            String[] prefixes = {"INFO:", "LOG:", "Event:", "Status:", "Alert:", "WARN:", "ERROR:"};
            msg = rand(prefixes) + " " + msg;
        }
        
        // Add suffixes sometimes
        if (random.nextDouble() < 0.15) {
            String[] suffixes = {"completed", "done", "finished", "OK", "pending", "failed", "success"};
            msg = msg + " - " + rand(suffixes);
        }
        
        // Add challenging edge cases
        if (random.nextDouble() < 0.1) {
            // Add ambiguous phrases that could be misclassified
            String[] ambiguous = {
                "Store supervisor authorized refund for customer",
                "Database connection lost during peak hours", 
                "Barcode scanner hardware failure detected",
                "Network connectivity issue resolved",
                "Printer malfunction reported by staff"
            };
            if (random.nextDouble() < 0.3) {
                msg = rand(ambiguous);
            }
        }
        
        return msg;
    }

    private static String generateLog(long index) {
        String label;
        long perClass = TOTAL_LOGS / 3;
        
        if (index < perClass) {
            label = "approval";
        } else if (index < perClass * 2) {
            label = "acknowledge";
        } else {
            label = "error";
        }

        String msg;
        
        // Add challenging edge cases (15% of logs) - increased for better training
        if (random.nextDouble() < 0.15) {
            String[] challengingCases = {
                "Store supervisor authorized refund for customer", // Should be approval
                "Database connection lost during peak hours", // Should be error  
                "Barcode scanner hardware failure detected", // Should be error
                "Network connectivity issue resolved", // Should be error
                "Printer malfunction reported by staff", // Should be error
                "Manager confirmed employee discount request", // Should be approval
                "System acknowledged payment processing", // Should be acknowledge
                "Customer confirmed receipt of refund", // Should be acknowledge
                "Staff verified inventory count", // Should be approval
                "Device acknowledged firmware update", // Should be acknowledge
                "Customer service rep verified account information", // Should be acknowledge
                "Support team acknowledged technical issue report", // Should be acknowledge
                "Server acknowledged maintenance window completion", // Should be acknowledge
                "System confirmed backup completion for database", // Should be acknowledge
                "Network connectivity issue resolved after router restart" // Should be error
            };
            msg = challengingCases[random.nextInt(challengingCases.length)];
        } else {
            switch (label) {
                case "approval": msg = makeApprovalLog(); break;
                case "acknowledge": msg = makeAckLog(); break;
                default: msg = makeErrorLog(); break;
            }
        }

        msg = addVariation(msg);
        msg = msg.replace("\"", "\"\"");

        // Sometimes add timestamp, sometimes don't
        if (random.nextDouble() < 0.5) {
            long now = System.currentTimeMillis();
            long offset = random.nextLong(1000L * 60 * 60 * 24 * 30);
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
                .format(new Date(now - offset));
            msg = String.format("[%s] %s", timestamp, msg);
        }

        return String.format("\"%s\",%s", msg, label);
    }

    public static void main(String[] args) {
        long start = System.currentTimeMillis();
        
        List<Long> indices = new ArrayList<>();
        for (long i = 0; i < TOTAL_LOGS; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices);
        
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE))) {
            writer.write("text,label\n");
            for (int i = 0; i < TOTAL_LOGS; i++) {
                writer.write(generateLog(indices.get(i)));
                writer.newLine();
                if ((i + 1) % CHUNK_SIZE == 0) {
                    System.out.printf("Generated %,d logs...%n", i + 1);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        long seconds = (System.currentTimeMillis() - start) / 1000;
        System.out.printf("✅ Done: %,d logs in %d sec%n", TOTAL_LOGS, seconds);
    }
}