import React from 'react';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Synthesize Personas</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Welcome to Synthesize Personas App
        </h1>

        <p className={styles.description}>
          Get started by uploading your data
        </p>

        <div className={styles.grid}>
          <a href="/api/hello" className={styles.card}>
            <h3>Upload data &rarr;</h3>
            <p>Upload your data to start synthesizing personas.</p>
          </a>
        </div>
      </main>

      <footer className={styles.footer}>
        <a
          href="https://vercel.com"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{' '}
          <img src="/vercel.svg" alt="Vercel Logo" className={styles.logo} />
        </a>
      </footer>
    </div>
  );
}